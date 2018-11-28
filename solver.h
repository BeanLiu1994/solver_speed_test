#pragma once

#define EIGEN_USE_MKL_ALL
#define USE_SUITESPARSE
#define USE_CUDA

#include <Eigen/Core>
#include <Eigen/Sparse>


enum EIGEN_SPSOLVE_METHOD
{
	METHOD_SimplicialLDLT,
#ifdef EIGEN_USE_MKL_ALL
	METHOD_PardisoLDLT,
	METHOD_UmfPackLU,
#endif
#ifdef USE_SUITESPARSE
	METHOD_ChomodSupernodalLLT,
#endif
#ifdef USE_CUDA
	METHOD_CUBLAS,
#endif
	METHOD_ConjugateGradient
};

constexpr EIGEN_SPSOLVE_METHOD defaultMethod =
#if defined USE_CUDA
METHOD_CUBLAS
#elif defined USE_SUITESPARSE
METHOD_ChomodSupernodalLLT
#elif defined EIGEN_USE_MKL_ALL
METHOD_PardisoLDLT
#else
METHOD_SimplicialLDLT
#endif
;

Eigen::MatrixXd _method_spsolve(const Eigen::SparseMatrix<double>& A, const Eigen::MatrixXd& b, EIGEN_SPSOLVE_METHOD method = defaultMethod);