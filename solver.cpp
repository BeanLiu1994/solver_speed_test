#include "solver.h"
#include "CuPtr.cuh"
#include <fstream>
#include <vector>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Sparse>

#ifdef EIGEN_USE_MKL_ALL
#include <Eigen/PardisoSupport>
#include <Eigen/UmfPackSupport>
#endif

#ifdef USE_SUITESPARSE
#include <Eigen/CholmodSupport>
#endif

#ifdef USE_CUDA
#include "CudaManager.h"
#include <cusolverSp.h>
#include <cusparse.h>
#endif

// methods
// *. Eigen 自带的 SimplicialLDLT (持平matlab)
// *. intel MKL pardiso 的 warpper: PardisoLDLT (第二快)
// *. SuiteSparse 的 warpper: UmfPackLU (不同问题时间不同好像,但慢于前二)
// *. SuiteSparse 的 warpper: CholmodSupernodalLLT (当前最快)
// 实际上根据问题规模速度也是不一样的,对于这种3w+的方阵应该如上所述

Eigen::MatrixXd cusparse_spsolve(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& b)
{
	Eigen::MatrixXd res;
	res.setZero(b.rows(), b.cols());

	// 此处实际应为 A.transpose() 但实际上A是对称矩阵,所以不处理了

	CuPtr<const double> csrVal(A.nonZeros(), A.valuePtr());
	CuPtr<const int> csrColInd(A.nonZeros(), A.innerIndexPtr());
	CuPtr<const int> csrRowPtr(A.cols() + 1, A.outerIndexPtr());

	cusolverSpHandle_t handle = NULL;
	cusparseHandle_t cusparseHandle = NULL;
	cudaStream_t stream = NULL;

	cusparseMatDescr_t descrA = 0;
	gpuErrchk(cusparseCreateMatDescr(&descrA));
	gpuErrchk(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	gpuErrchk(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
	double tol = 0.01;
	int reorder = 1;
	int singularity;

	for (int i = 0; i < b.cols(); ++i)
	{
		const Eigen::VectorXd bi = b.col(i);
		Eigen::VectorXd x;
		x.setZero(bi.rows(), bi.cols());

		CuPtr<const double> Dbi(bi.size(), bi.data());
		CuPtr<double> Dx(x.size(), x.data());

		cusolverStatus_t cuStatus = cusolverSpCreate(&handle);
		if (cuStatus == CUSOLVER_STATUS_SUCCESS)
		{
			gpuErrchk(cusparseCreate(&cusparseHandle));
			gpuErrchk(cudaStreamCreate(&stream));
			gpuErrchk(cusolverSpSetStream(handle, stream));
			gpuErrchk(cusparseSetStream(cusparseHandle, stream));
			cusolverStatus_t judge = cusolverSpDcsrlsvchol(
				handle, A.rows(), A.nonZeros(), descrA,
				csrVal(), csrRowPtr(), csrColInd(), Dbi(),
				tol, reorder, Dx(), &singularity
			);
			if (judge != CUSOLVER_STATUS_SUCCESS)
			{
				std::cout << "cuSolver failed to solve. error code: " << judge << std::endl;
				throw std::runtime_error("cusparse failed.");
			}
			Dx.GetResult();
			cusparseDestroy(cusparseHandle);
			cudaStreamDestroy(stream);
		}
		cusolverSpDestroy(handle);
		res.col(i) = x;
	}
	cusparseDestroyMatDescr(descrA);

	return res;
}

Eigen::MatrixXd _method_spsolve(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& b, EIGEN_SPSOLVE_METHOD method)
{
	switch (method)
	{
	case METHOD_SimplicialLDLT:
	{
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Solver(A);
		if (Solver.info() != Eigen::Success) {
			throw std::runtime_error("solver build failed.");
		}
		return Solver.solve(b);
	}break;
#ifdef EIGEN_USE_MKL_ALL
	case METHOD_PardisoLDLT:
	{
		Eigen::PardisoLDLT<Eigen::SparseMatrix<double>> Solver(A);
		if (Solver.info() != Eigen::Success) {
			throw std::runtime_error("solver build failed.");
		}
		return Solver.solve(b);
	}break;
	case METHOD_UmfPackLU:
	{
		Eigen::UmfPackLU<Eigen::SparseMatrix<double>> Solver(A);
		if (Solver.info() != Eigen::Success) {
			throw std::runtime_error("solver build failed.");
		}
		return Solver.solve(b);
	}break;
#endif
#ifdef USE_SUITESPARSE
	case METHOD_ChomodSupernodalLLT:
	{
		Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> Solver(A);
		if (Solver.info() != Eigen::Success) {
			throw std::runtime_error("solver build failed.");
		}
		return Solver.solve(b);
	}break;
#endif
#ifdef USE_CUDA	
	case METHOD_CUBLAS:
	{
		return cusparse_spsolve(A, b);
	}break;
#endif
	case METHOD_ConjugateGradient:
	{
		Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> Solver(A);
		if (Solver.info() != Eigen::Success) {
			throw std::runtime_error("solver build failed.");
		}
		return Solver.solve(b);
	}break;
	default:
		return Eigen::MatrixXd();
	}
}


template<typename _ty = std::chrono::microseconds>
class Timer
{
public:
	Timer(bool startNow = true)
	{
		if (startNow) StartTimer();
	}
	void StartTimer(const std::string& _printThis = std::string())
	{
		if (!_printThis.empty()) std::cout << _printThis << std::endl;
		start = std::chrono::system_clock::now();
	}
	double EndTimer(const std::string& _printThis, bool restartFlag)
	{
		auto seconds = EndTimer(_printThis);
		if (restartFlag)
			StartTimer();
		return seconds;
	}
	double EndTimer(const std::string& _printThis = std::string())
	{
		end = std::chrono::system_clock::now();
		duration = std::chrono::duration_cast<_ty>(end - start);
		seconds = (double(duration.count()) * _ty::period::num / _ty::period::den);
		std::string s = " elapsed time:  " + std::to_string(seconds) + "s";
		if (!_printThis.empty()) std::cout << _printThis << s << std::endl;
		//else std::cout << s << std::endl;
		return seconds;
	}
private:
	std::chrono::time_point<std::chrono::system_clock> start, end;
	_ty duration;
	double seconds;
};

Eigen::SparseMatrix<double> LoadSP(const std::string& path)
{
	int rows, cols, r, c;
	double v;
	std::ifstream file(path);
	if (!file)
		return Eigen::SparseMatrix<double>();
	file >> rows >> cols;

	Eigen::SparseMatrix<double> ret(rows, cols);
	while (file >> r >> c >> v)
	{
		ret.insert(r, c) = v;
	}
	ret.makeCompressed();
	return ret;
}

Eigen::VectorXd LoadVec(const std::string& path)
{
	double v;
	std::ifstream file(path);
	if (!file)
		return Eigen::VectorXd();

	std::vector<double> data;
	while (file >> v)
	{
		data.push_back(v);
	}
	Eigen::VectorXd ret;
	ret.setZero(data.size());
	for (int i = 0; i < ret.size(); ++i)
	{
		ret(i) = data[i];
	}
	return ret;
}

std::vector<double> DoTest(const Eigen::SparseMatrix<double>& A_full, const Eigen::VectorXd& b_full, int edge_size)
{
	std::cout << "resizing matrix for test" << std::endl;
	Eigen::SparseMatrix<double> A = A_full.topLeftCorner(edge_size, edge_size);
	A.makeCompressed();
	Eigen::VectorXd b = b_full.topRows(edge_size);

	std::cout << "Size : " << edge_size << " . Beginning..." << std::endl;

	std::vector<double> timing;

	Timer<> timer;
	auto x1 = _method_spsolve(A, b, METHOD_ChomodSupernodalLLT);
	timing.push_back(timer.EndTimer("suitesparse-chol: "));
	double Level = x1.cwiseAbs().mean();

	timer.StartTimer();
	auto x2 = _method_spsolve(A, b, METHOD_PardisoLDLT);
	timing.push_back(timer.EndTimer("mkl-pardiso: "));
	std::cout << "error percent: " <<  (x2 - x1).cwiseAbs().mean() / Level << std::endl;

	timer.StartTimer();
	auto x3 = _method_spsolve(A, b, METHOD_SimplicialLDLT);
	timing.push_back(timer.EndTimer("eigen-ldlt: "));
	std::cout << "error percent: " << (x3 - x1).cwiseAbs().mean() / Level << std::endl;

	timer.StartTimer();
	auto x4 = _method_spsolve(A, b, METHOD_UmfPackLU);
	timing.push_back(timer.EndTimer("suitesparse-umf: "));
	std::cout << "error percent: " << (x4 - x1).cwiseAbs().mean() / Level << std::endl;

	timer.StartTimer();
	auto x5 = _method_spsolve(A, b, METHOD_CUBLAS);
	timing.push_back(timer.EndTimer("gpu-cublas: "));
	std::cout << "error percent: " << (x5 - x1).cwiseAbs().mean() / Level << std::endl;

	timer.StartTimer();
	auto x6 = _method_spsolve(A, b, METHOD_ConjugateGradient);
	timing.push_back(timer.EndTimer("eigen-CGiterative: "));
	std::cout << "error percent: " << (x6 - x1).cwiseAbs().mean() / Level << std::endl;

	std::cout << std::endl;

	return timing;
}

void WriteResult(std::string path, std::vector<int> sizeLists, std::vector<std::vector<double>> time)
{
	std::ofstream file(path);
	if (!file)
		return;
	for (int i = 0; i < sizeLists.size(); ++i)
	{
		file << sizeLists[i];
		for (int j = 0; j < time[i].size(); ++j)
			file << "," << time[i][j];
		file << std::endl;
	}
	return;
}

int main()
{
	std::vector<int> sizeLists
	{
		10, 100, 300, 500, 700, 1000, 2600, 5000, 7400, 
		10000, 15000, 20000, 25000, 30000
		, 40000, 50000, 60000, 70000, 80000, 90000
	};
	Eigen::SparseMatrix<double> A = LoadSP("spA");
	Eigen::VectorXd b = LoadVec("b");

	std::cout << A.rows() << " and " << A.cols() << "and non-zero " << A.nonZeros() << std::endl;
	std::cout << b.rows() << std::endl;

	// for some methods' initialization
	DoTest(A, b, 6);

	cudaInitializer::Init();

	std::vector<std::vector<double>> time;
	for (auto m : sizeLists)
	{
		time.push_back(DoTest(A, b, m));
	}

	WriteResult("out.csv", sizeLists, time);

	return 0;
}