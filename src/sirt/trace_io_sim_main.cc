#include <chrono>
#include "mpi.h"
#include "trace_h5io.h"
#include "disp_comm_mpi.h"
#include "tclap/CmdLine.h"

class TraceRuntimeConfig {
  public:
    float *data_ = nullptr;
    float *theta_ = nullptr;

    const float kPI = 3.14159265358979f;
    int kNProjections;
    int kNSinograms;
    int kNColumns;
    std::string kReconOutputPath;
    int kMPIOXferFlag;

    TraceRuntimeConfig(int argc, char **argv, int rank, int size){
      try
      {
        TCLAP::CmdLine cmd("TRACE I/O Simulator", ' ', "0.01");
        TCLAP::ValueArg<int> argNProjections(
          "p", "nProjections", "Number of projections", true, 0, "int");
        TCLAP::ValueArg<int> argNSinograms(
          "s", "nSinograms", "Number of sinograms", true, 0, "int");
        TCLAP::ValueArg<int> argNColumns(
          "c", "nColumns", "Number of columns", true, 0, "int");
        TCLAP::ValueArg<std::string> argReconOutputPath(
          "o", "reconOutputPath", "Output file path for reconstructed image (hdf5)",
          true, "", "string");

        std::vector<int> allowed_xfer_flags_vals;
        allowed_xfer_flags_vals.push_back(0);
        allowed_xfer_flags_vals.push_back(1);
        TCLAP::ValuesConstraint<int> allowed_xfer_flags(allowed_xfer_flags_vals);
        TCLAP::ValueArg<int> argMPIOXferFlag(
          "x", "mpioXferFlag", "MPI-IO transfer flag (Independent=0, Collective=1)",
          true, 1, &allowed_xfer_flags);

        cmd.add(argReconOutputPath);
        cmd.add(argMPIOXferFlag);
        cmd.add(argNColumns);
        cmd.add(argNSinograms);
        cmd.add(argNProjections);

        cmd.parse(argc, argv);
        kNProjections = argNProjections.getValue();
        kNSinograms = argNSinograms.getValue();
        kNColumns = argNColumns.getValue();
        kReconOutputPath = argReconOutputPath.getValue();
        kMPIOXferFlag = argMPIOXferFlag.getValue();

        if(rank==0)
        {
          std::cout << "MPI rank:"<< rank << "; MPI size:" << size << std::endl;
          std::cout << "Number of projections=" << kNProjections << std::endl;
          std::cout << "Number of sinograms=" << kNSinograms << std::endl;
          std::cout << "Number of columns=" << kNColumns << std::endl;
          std::cout << "Output file path=" << kReconOutputPath << std::endl;
          std::cout << "MPI-IO transfer flag (Independent=0, Collective=1)=" << kMPIOXferFlag << std::endl;
        }
      }
      catch (TCLAP::ArgException &e)
      {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
      }
    }
};


class TDataMock
{
  private:
    float *data_ = nullptr;
    float *theta_ = nullptr;
    const float kPI = 3.14159265358979f;
    int beg_index_;
    int num_projs_;
    int num_slices_;
    int num_cols_;
    int num_iter_;
    int num_threads_;

  public:
    TDataMock(
        int beg_index, 
        int num_projs, 
        int num_slices, 
        int num_cols,
        int num_iter,
        int num_threads):
      beg_index_{beg_index},
      num_projs_{num_projs}, 
      num_slices_{num_slices},
      num_cols_{num_cols},
      num_iter_{num_iter},
      num_threads_{num_threads}
    {
      data_ = new float[num_projs_*num_slices_*num_cols_];
      theta_ = new float[num_cols_];
    }

    ~TDataMock()
    {
      delete [] data_;
      // delete [] theta_;
    }

    void GenParData(float val, int rank)
    {
      for(int i=0; i<num_projs_; ++i)
        for(int j=0; j<num_slices_; ++j)
          for(int k=0; k<num_cols_; ++k)
            data_[i*num_slices_*num_cols_+j*num_cols_+k] = 
              rank + i*num_slices_*num_cols_ + j*num_cols_ +k;
    }
    void GenProjTheta(float beg, float end)
    {
      float rate = (end-beg)/num_cols_;
      for(int i=0; i<num_cols_; ++i)
        theta_[i] = (beg+i*rate)*kPI/180.;
    }

    int beg_index() const { return beg_index_; }
    int num_projs() const { return num_projs_; }
    int num_slices() const { return num_slices_; }
    int num_cols() const { return num_cols_; }
    float* data() const { return data_; }
    float* theta() const { return theta_; }
    int num_iter() const { return num_iter_; }
    int num_threads() const { return num_threads_; };
};



int main(int argc, char **argv)
{
  auto tot_beg_time = std::chrono::high_resolution_clock::now();
  
  /* Initiate middleware's communication layer */
  DISPCommBase<float> *comm =
        new DISPCommMPI<float>(&argc, &argv);
  TraceRuntimeConfig config(argc, argv, comm->rank(), comm->size());

  int beg_index, n_blocks;
  trace_io::DistributeSlices(
      comm->rank(), comm->size(), 
      config.kNSinograms, beg_index, n_blocks);

  auto dg_beg_time = std::chrono::high_resolution_clock::now();
  TDataMock mock_data(
      beg_index,              /// Beginning index
      config.kNProjections,   /// Number of projections
      n_blocks,               /// Number of sinograms
      config.kNColumns,       /// Number of columns
      0, 1);                  /// Iteration and thread counts
  mock_data.GenParData(0., comm->rank());
  mock_data.GenProjTheta(0., 360.);
  std::chrono::duration<double> dg_time = 
    std::chrono::high_resolution_clock::now()-dg_beg_time; 


  /* Setup metadata data structure */
  // INFO: TraceMetadata destructor frees theta->data!
  // TraceMetadata internally creates reconstruction object
  TraceMetadata trace_metadata(
      static_cast<float *>(mock_data.theta()),  /// float const *theta,
      0,                                        /// int const proj_id,
      mock_data.beg_index(),                    /// int const slice_id,
      0,                                        /// int const col_id,
      mock_data.num_slices(),                   /// int const num_tot_slices,
      mock_data.num_projs(),                    /// int const num_projs,
      mock_data.num_slices(),                   /// int const num_slices,
      mock_data.num_cols(),                     /// int const num_cols,
      mock_data.num_cols(),                     /// int const num_grids,
      0.);                                      /// float const center

  auto &recon = trace_metadata.recon();
  for(size_t i=0; i<recon.count(); i++)
    recon[i]=comm->rank()+i;

  /* Setup output file metadata */
  trace_io::H5Metadata d_metadata = {
    config.kReconOutputPath,
    "/recon",
    3,
    new hsize_t[3]
    {static_cast<hsize_t>(config.kNProjections),
     static_cast<hsize_t>(config.kNSinograms),
     static_cast<hsize_t>(config.kNColumns)},
    config.kNSinograms*config.kNColumns*config.kNColumns*sizeof(float)
  };

  /* Write reconstructed data to disk */
  H5FD_mpio_xfer_t mpio_xfer_flag = (config.kMPIOXferFlag==0) ? 
    H5FD_MPIO_INDEPENDENT : H5FD_MPIO_COLLECTIVE;
  auto write_beg_time = std::chrono::high_resolution_clock::now();
  trace_io::WriteRecon(
      trace_metadata, d_metadata, 
      config.kReconOutputPath, 
      "/recon",
      mpio_xfer_flag);
  std::chrono::duration<double> write_time = 
    std::chrono::high_resolution_clock::now()-write_beg_time; 

  std::chrono::duration<double> tot_time = 
    std::chrono::high_resolution_clock::now()-tot_beg_time; 

  std::cout << "Write time=" << write_time.count() <<
    "; Total time=" << tot_time.count() <<
    "; Data Generation time=" << dg_time.count() << std::endl;

  delete comm;
}

