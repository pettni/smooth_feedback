// Copyright (C) 2022 Petter Nilsson. MIT License.

#pragma once

/**
 * @file
 * @brief Formulate optimal control problem as a nonlinear program
 */

#include <Eigen/Core>
#include <smooth/concepts/lie_group.hpp>

#include "collocation/mesh.hpp"
#include "collocation/mesh_function.hpp"
#include "nlp.hpp"
#include "ocp.hpp"

namespace smooth::feedback {

// \cond
namespace detail {

/// @brief Variable and constraint structure of an OCP NLP
auto ocp_nlp_structure(const FlatOCPType auto & ocp, const MeshType auto & mesh)
{
  std::size_t N = mesh.N_colloc();

  // variable layout
  std::array<std::size_t, 4> var_len{
    1,                 // tf
    ocp.Nq,            // integrals
    ocp.Nx * (N + 1),  // states
    ocp.Nu * N,        // inputs
  };

  // constraint layout
  std::array<std::size_t, 4> con_len{
    ocp.Nx * N,   // derivatives
    ocp.Nq,       // other integrals
    ocp.Ncr * N,  // running constraints
    ocp.Nce,      // end constraints
  };

  std::array<std::size_t, 5> var_beg{0};
  std::partial_sum(var_len.begin(), var_len.end(), var_beg.begin() + 1);

  std::array<std::size_t, 5> con_beg{0};
  std::partial_sum(con_len.begin(), con_len.end(), con_beg.begin() + 1);

  return std::make_tuple(var_beg, var_len, con_beg, con_len);
}

/**
 * @brief NLP representing an OCP.
 *
 * @note This class allocates matrices internally and returns references to those.
 */
template<FlatOCPType Ocp, MeshType Mesh, diff::Type DT = diff::Type::Default>
class OCPNLP
{
private:
  static constexpr auto Nx = std::decay_t<Ocp>::Nx;
  static constexpr auto Nu = std::decay_t<Ocp>::Nu;
  static constexpr auto Nq = std::decay_t<Ocp>::Nq;

  Ocp ocp_;        // optimal control problem
  Mesh mesh_;      // discretization mesh
  std::size_t N_;  // number of collocation points in mesh

  std::size_t tfvar_B, qvar_B, xvar_B, uvar_B, n_;  // variable start indices
  std::size_t tfvar_L, qvar_L, xvar_L, uvar_L;      // variable lengths

  std::size_t dcon_B, qcon_B, crcon_B, cecon_B, m_;  // constraint start indices
  std::size_t dcon_L, qcon_L, crcon_L, cecon_L;      // constraint lengths

  // scaling
  double w_scaling_{1};

  // variable and constraint bounds
  Eigen::VectorXd xl_, xu_, gl_, gu_;

  // allocated return arguments
  Eigen::VectorXd g_;
  Eigen::SparseMatrix<double> df_dx_, dg_dx_, d2f_dx2_, d2g_dx2_;

  // allocated computation
  MeshValue<0> dyn_out0_, int_out0_, cr_out0_;
  MeshValue<1> dyn_out1_, int_out1_, cr_out1_;
  MeshValue<2> dyn_out2_, int_out2_, cr_out2_;

public:
  /// @brief Constructor
  template<typename OcpArg, typename MeshArg>
  OCPNLP(OcpArg && ocp, MeshArg && mesh)
      : ocp_(std::forward<OcpArg>(ocp)), mesh_(std::forward<MeshArg>(mesh)), N_(mesh_.N_colloc())
  {
    const auto [var_beg, var_len, con_beg, con_len] = detail::ocp_nlp_structure(ocp_, mesh_);

    tfvar_B = var_beg[0];
    qvar_B  = var_beg[1];
    xvar_B  = var_beg[2];
    uvar_B  = var_beg[3];
    n_      = var_beg[4];

    tfvar_L = var_len[0];
    qvar_L  = var_len[1];
    xvar_L  = var_len[2];
    uvar_L  = var_len[3];

    dcon_B  = con_beg[0];
    qcon_B  = con_beg[1];
    crcon_B = con_beg[2];
    cecon_B = con_beg[3];
    m_      = con_beg[4];

    dcon_L  = con_len[0];
    qcon_L  = con_len[1];
    crcon_L = con_len[2];
    cecon_L = con_len[3];

    // Mesh weight
    double max_weight = 1e-6;
    for (auto w : mesh_.all_weights()) { max_weight = std::max(max_weight, w); }
    w_scaling_ = 1. / max_weight;

    // VARIABLE BOUNDS

    xl_.setConstant(n_, -std::numeric_limits<double>::infinity());
    xl_.segment(tfvar_B, tfvar_L).setZero();  // tf lower bounded by zero
    xu_.setConstant(n_, std::numeric_limits<double>::infinity());

    // CONSTRAINT BOUNDS

    gl_.resize(m_);
    gu_.resize(m_);

    // derivative constraints are equalities
    gl_.segment(dcon_B, dcon_L).setZero();
    gu_.segment(dcon_B, dcon_L).setZero();

    // integral constraints are equalities
    gl_.segment(qcon_B, qcon_L).setZero();
    gu_.segment(qcon_B, qcon_L).setZero();

    // running constraints (scaled by quadrature weights)
    gl_.segment(crcon_B, crcon_L) = ocp.crl.replicate(N_, 1);
    gu_.segment(crcon_B, crcon_L) = ocp.cru.replicate(N_, 1);
    for (const auto & [i, w] : utils::zip(std::views::iota(0u, N_), mesh_.all_weights())) {
      gl_.segment(crcon_B + i * ocp_.Ncr, ocp_.Ncr) *= w_scaling_ * w;
      gu_.segment(crcon_B + i * ocp_.Ncr, ocp_.Ncr) *= w_scaling_ * w;
    }

    // end constraints
    gl_.segment(cecon_B, cecon_L) = ocp.cel;
    gu_.segment(cecon_B, cecon_L) = ocp.ceu;

    // allocate output args
    g_.setZero(m_);

    df_dx_.resize(1, n_);
    df_dx_.reserve(Eigen::VectorXi::Constant(n_, 1));

    d2f_dx2_.resize(n_, n_);
    dg_dx_.resize(m_, n_);
    d2g_dx2_.resize(n_, n_);

    /// @todo Allocate nnz's to speed up first call?
  }

  std::size_t n() const { return n_; }
  std::size_t m() const { return m_; }
  const Eigen::VectorXd & xl() const { return xl_; }
  const Eigen::VectorXd & xu() const { return xu_; }

  double f(const Eigen::Ref<const Eigen::VectorXd> x) const
  {
    assert(static_cast<std::size_t>(x.size()) == n_);

    const auto x0var_B = xvar_B;
    const auto xfvar_B = xvar_B + xvar_L - Nx;

    const double tf                    = x(tfvar_B);
    const Eigen::Vector<double, Nx> x0 = x.segment(x0var_B, Nx);
    const Eigen::Vector<double, Nx> xf = x.segment(xfvar_B, Nx);
    const Eigen::Vector<double, Nq> q  = x.segment(qvar_B, qvar_L);

    return ocp_.theta(tf, x0, xf, q);
  }

  const Eigen::SparseMatrix<double> & df_dx(const Eigen::Ref<const Eigen::VectorXd> x)
  {
    assert(static_cast<std::size_t>(x.size()) == n_);

    const auto x0var_B = xvar_B;
    const auto xfvar_B = xvar_B + xvar_L - Nx;

    const double tf                    = x(tfvar_B);
    const Eigen::Vector<double, Nx> x0 = x.segment(x0var_B, Nx);
    const Eigen::Vector<double, Nx> xf = x.segment(xfvar_B, Nx);
    const Eigen::Vector<double, Nq> q  = x.segment(qvar_B, qvar_L);

    const auto & [fval, dfval] = diff::dr<1, DT>(ocp_.theta, wrt(tf, x0, xf, q));

    set_zero(df_dx_);

    block_add(df_dx_, 0, tfvar_B, dfval.middleCols(0, 1));           // df / dtf
    block_add(df_dx_, 0, x0var_B, dfval.middleCols(1, Nx));          // df / dx0
    block_add(df_dx_, 0, xfvar_B, dfval.middleCols(1 + Nx, Nx));     // df / dxf
    block_add(df_dx_, 0, qvar_B, dfval.middleCols(1 + 2 * Nx, Nq));  // df / dq

    df_dx_.makeCompressed();
    return df_dx_;
  }

  const Eigen::SparseMatrix<double> & d2f_dx2(Eigen::Ref<const Eigen::VectorXd> x)
  {
    assert(static_cast<std::size_t>(x.size()) == n_);

    const auto x0var_B = xvar_B;
    const auto xfvar_B = xvar_B + xvar_L - Nx;

    const double tf                    = x(tfvar_B);
    const Eigen::Vector<double, Nx> x0 = x.segment(x0var_B, Nx);
    const Eigen::Vector<double, Nx> xf = x.segment(xfvar_B, Nx);
    const Eigen::Vector<double, Nq> q  = x.segment(qvar_B, qvar_L);

    const auto & [fval, dfval, d2fval] = diff::dr<2, DT>(ocp_.theta, wrt(tf, x0, xf, q));

    set_zero(d2f_dx2_);

    // clang-format off
    block_add(d2f_dx2_, tfvar_B, tfvar_B, d2fval.block(         0,          0,  1,  1), 1, true);  // tftf
    block_add(d2f_dx2_, tfvar_B, x0var_B, d2fval.block(         0,          1,  1, Nx), 1, true);  // tfx0
    block_add(d2f_dx2_, tfvar_B, xfvar_B, d2fval.block(         0,     1 + Nx,  1, Nx), 1, true);  // tfxf
    block_add(d2f_dx2_, tfvar_B,  qvar_B, d2fval.block(         0, 1 + 2 * Nx,  1, Nq), 1, true);  // tfq

    block_add(d2f_dx2_, x0var_B, x0var_B, d2fval.block(         1,          1, Nx, Nx), 1, true);  // x0x0
    block_add(d2f_dx2_, x0var_B, xfvar_B, d2fval.block(         1,     1 + Nx, Nx, Nx), 1, true);  // x0xf
    block_add(d2f_dx2_, x0var_B,  qvar_B, d2fval.block(         1, 1 + 2 * Nx, Nx, Nq), 1, true);  // x0q

    block_add(d2f_dx2_, xfvar_B, xfvar_B, d2fval.block(    1 + Nx,     1 + Nx, Nx, Nx), 1, true);  // xfxf
    block_add(d2f_dx2_, xfvar_B,  qvar_B, d2fval.block(    1 + Nx, 1 + 2 * Nx, Nx, Nq), 1, true);  // xfq

    block_add(d2f_dx2_,  qvar_B,  qvar_B, d2fval.block(1 + 2 * Nx, 1 + 2 * Nx, Nq, Nq), 1, true);  // qq
    // clang-format on

    d2f_dx2_.makeCompressed();
    return d2f_dx2_;
  }

  const Eigen::VectorXd & g(const Eigen::Ref<const Eigen::VectorXd> x)
  {
    assert(static_cast<std::size_t>(x.size()) == n_);

    const auto x0var_B = xvar_B;
    const auto xfvar_B = xvar_B + xvar_L - Nx;

    const double t0                    = 0;
    const double tf                    = x(tfvar_B);
    const Eigen::Vector<double, Nx> x0 = x.segment(x0var_B, Nx);
    const Eigen::Vector<double, Nx> xf = x.segment(xfvar_B, Nx);
    const Eigen::Vector<double, Nq> q  = x.segment(qvar_B, qvar_L);

    const Eigen::Map<const Eigen::Matrix<double, Nx, -1>> X(x.data() + xvar_B, Nx, N_ + 1);
    const Eigen::Map<const Eigen::Matrix<double, Nu, -1>> U(x.data() + uvar_B, Nu, N_);

    mesh_dyn<0>(dyn_out0_, mesh_, ocp_.f, t0, tf, X.colwise(), U.colwise());
    mesh_integrate<0>(int_out0_, mesh_, ocp_.g, t0, tf, X.colwise(), U.colwise());
    mesh_eval<0>(cr_out0_, mesh_, ocp_.cr, t0, tf, X.colwise(), U.colwise(), true);

    g_.segment(dcon_B, dcon_L)   = w_scaling_ * dyn_out0_.F;
    g_.segment(qcon_B, qcon_L)   = w_scaling_ * (int_out0_.F - q);
    g_.segment(crcon_B, crcon_L) = w_scaling_ * cr_out0_.F;
    g_.segment(cecon_B, cecon_L) = ocp_.ce(tf, x0, xf, q);

    return g_;
  }
  const Eigen::VectorXd & gl() const { return gl_; }
  const Eigen::VectorXd & gu() const { return gu_; }
  const Eigen::SparseMatrix<double> & dg_dx(const Eigen::Ref<const Eigen::VectorXd> x)
  {
    assert(static_cast<std::size_t>(x.size()) == n_);

    const auto x0var_B = xvar_B;
    const auto xfvar_B = xvar_B + xvar_L - Nx;

    const double t0                    = 0;
    const double tf                    = x(tfvar_B);
    const Eigen::Vector<double, Nx> x0 = x.segment(x0var_B, Nx);
    const Eigen::Vector<double, Nx> xf = x.segment(xfvar_B, Nx);
    const Eigen::Vector<double, Nq> q  = x.segment(qvar_B, qvar_L);

    const Eigen::Map<const Eigen::Matrix<double, Nx, -1>> X(x.data() + xvar_B, Nx, N_ + 1);
    const Eigen::Map<const Eigen::Matrix<double, Nu, -1>> U(x.data() + uvar_B, Nu, N_);

    mesh_dyn<1, DT>(dyn_out1_, mesh_, ocp_.f, t0, tf, X.colwise(), U.colwise());
    mesh_integrate<1, DT>(int_out1_, mesh_, ocp_.g, t0, tf, X.colwise(), U.colwise());
    mesh_eval<1, DT>(cr_out1_, mesh_, ocp_.cr, t0, tf, X.colwise(), U.colwise(), true);
    const auto & [ceval, dceval] = diff::dr<1, DT>(ocp_.ce, wrt(tf, x0, xf, q));

    dyn_out1_.dF.makeCompressed();
    int_out1_.dF.makeCompressed();
    cr_out1_.dF.makeCompressed();

    set_zero(dg_dx_);

    // dynamics constraint
    block_add(dg_dx_, dcon_B, tfvar_B, dyn_out1_.dF.middleCols(1, 1), w_scaling_);
    block_add(dg_dx_, dcon_B, xvar_B, dyn_out1_.dF.middleCols(2, xvar_L), w_scaling_);
    block_add(dg_dx_, dcon_B, uvar_B, dyn_out1_.dF.middleCols(2 + xvar_L, uvar_L), w_scaling_);

    // integral constraint
    block_add(dg_dx_, qcon_B, tfvar_B, int_out1_.dF.middleCols(1, 1), w_scaling_);
    block_add(dg_dx_, qcon_B, xvar_B, int_out1_.dF.middleCols(2, xvar_L), w_scaling_);
    block_add(dg_dx_, qcon_B, uvar_B, int_out1_.dF.middleCols(2 + xvar_L, uvar_L), w_scaling_);
    block_add_identity(dg_dx_, qcon_B, qvar_B, qvar_L, -w_scaling_);

    // running constraint
    block_add(dg_dx_, crcon_B, tfvar_B, cr_out1_.dF.middleCols(1, 1), w_scaling_);
    block_add(dg_dx_, crcon_B, xvar_B, cr_out1_.dF.middleCols(2, xvar_L), w_scaling_);
    block_add(dg_dx_, crcon_B, uvar_B, cr_out1_.dF.middleCols(2 + xvar_L, uvar_L), w_scaling_);

    // end constraint
    block_add(dg_dx_, cecon_B, tfvar_B, dceval.middleCols(0, 1));
    block_add(dg_dx_, cecon_B, xvar_B, dceval.middleCols(1, Nx));
    block_add(dg_dx_, cecon_B, xvar_B + xvar_L - Nx, dceval.middleCols(1 + Nx, Nx));
    block_add(dg_dx_, cecon_B, qvar_B, dceval.middleCols(1 + 2 * Nx, Nq));

    dg_dx_.makeCompressed();
    return dg_dx_;
  }

  const Eigen::SparseMatrix<double> &
  d2g_dx2(const Eigen::Ref<const Eigen::VectorXd> x, const Eigen::Ref<const Eigen::VectorXd> lambda)
  {
    assert(static_cast<std::size_t>(x.size()) == n_);
    assert(static_cast<std::size_t>(lambda.size()) == m_);

    const auto x0var_B = xvar_B;
    const auto xfvar_B = xvar_B + xvar_L - Nx;

    const double t0                    = 0;
    const double tf                    = x(tfvar_B);
    const Eigen::Vector<double, Nx> x0 = x.segment(x0var_B, Nx);
    const Eigen::Vector<double, Nx> xf = x.segment(xfvar_B, Nx);
    const Eigen::Vector<double, Nq> q  = x.segment(qvar_B, qvar_L);

    const Eigen::Map<const Eigen::Matrix<double, Nx, -1>> X(x.data() + xvar_B, Nx, N_ + 1);
    const Eigen::Map<const Eigen::Matrix<double, Nu, -1>> U(x.data() + uvar_B, Nu, N_);

    dyn_out2_.lambda = lambda.segment(dcon_B, dcon_L);
    int_out2_.lambda = lambda.segment(qcon_B, qcon_L);
    cr_out2_.lambda  = lambda.segment(crcon_B, crcon_L);
    mesh_dyn<2, DT>(dyn_out2_, mesh_, ocp_.f, t0, tf, X.colwise(), U.colwise());
    mesh_integrate<2, DT>(int_out2_, mesh_, ocp_.g, t0, tf, X.colwise(), U.colwise());
    mesh_eval<2, DT>(cr_out2_, mesh_, ocp_.cr, t0, tf, X.colwise(), U.colwise(), true);
    const auto & [ceval, dceval, d2ceval] = diff::dr<2, DT>(ocp_.ce, wrt(tf, x0, xf, q));

    dyn_out2_.dF.makeCompressed();
    int_out2_.dF.makeCompressed();
    cr_out2_.dF.makeCompressed();

    dyn_out2_.d2F.makeCompressed();
    int_out2_.d2F.makeCompressed();
    cr_out2_.d2F.makeCompressed();

    set_zero(d2g_dx2_);

    // clang-format off
    block_add(d2g_dx2_, tfvar_B, tfvar_B, dyn_out2_.d2F.block(1, 1, 1, 1), w_scaling_, true);      // tftf
    block_add(d2g_dx2_, tfvar_B, tfvar_B, int_out2_.d2F.block(1, 1, 1, 1), w_scaling_, true);      // tftf
    block_add(d2g_dx2_, tfvar_B, tfvar_B,  cr_out2_.d2F.block(1, 1, 1, 1), w_scaling_, true);      // tftf

    block_add(d2g_dx2_, tfvar_B, xvar_B, dyn_out2_.d2F.block(1, 2, 1, xvar_L), w_scaling_, true);  // tfx
    block_add(d2g_dx2_, tfvar_B, xvar_B, int_out2_.d2F.block(1, 2, 1, xvar_L), w_scaling_, true);  // tfx
    block_add(d2g_dx2_, tfvar_B, xvar_B,  cr_out2_.d2F.block(1, 2, 1, xvar_L), w_scaling_, true);  // tfx

    block_add(d2g_dx2_, tfvar_B, uvar_B, dyn_out2_.d2F.block(1, 2 + xvar_L, 1, uvar_L), w_scaling_, true);  // tfu
    block_add(d2g_dx2_, tfvar_B, uvar_B, int_out2_.d2F.block(1, 2 + xvar_L, 1, uvar_L), w_scaling_, true);  // tfu
    block_add(d2g_dx2_, tfvar_B, uvar_B,  cr_out2_.d2F.block(1, 2 + xvar_L, 1, uvar_L), w_scaling_, true);  // tfu

    block_add(d2g_dx2_, xvar_B, xvar_B, dyn_out2_.d2F.block(2,          2, xvar_L, xvar_L), w_scaling_, true);  // xx
    block_add(d2g_dx2_, xvar_B, xvar_B, int_out2_.d2F.block(2,          2, xvar_L, xvar_L), w_scaling_, true);  // xx
    block_add(d2g_dx2_, xvar_B, xvar_B,  cr_out2_.d2F.block(2,          2, xvar_L, xvar_L), w_scaling_, true);  // xx

    block_add(d2g_dx2_, xvar_B, uvar_B, dyn_out2_.d2F.block(2, 2 + xvar_L, xvar_L, uvar_L), w_scaling_, true);  // xu
    block_add(d2g_dx2_, xvar_B, uvar_B, int_out2_.d2F.block(2, 2 + xvar_L, xvar_L, uvar_L), w_scaling_, true);  // xu
    block_add(d2g_dx2_, xvar_B, uvar_B,  cr_out2_.d2F.block(2, 2 + xvar_L, xvar_L, uvar_L), w_scaling_, true);  // xu

    block_add(d2g_dx2_, uvar_B, uvar_B, dyn_out2_.d2F.block(2 + xvar_L, 2 + xvar_L, uvar_L, uvar_L), w_scaling_, true);  // uu
    block_add(d2g_dx2_, uvar_B, uvar_B, int_out2_.d2F.block(2 + xvar_L, 2 + xvar_L, uvar_L, uvar_L), w_scaling_, true);  // uu
    block_add(d2g_dx2_, uvar_B, uvar_B,  cr_out2_.d2F.block(2 + xvar_L, 2 + xvar_L, uvar_L, uvar_L), w_scaling_, true);  // uu
    // clang-format on

    for (auto j = 0u; j < ocp_.Nce; ++j) {
      const auto b0 = (1 + 2 * Nx + ocp_.Nq) * j;
      // clang-format off
      block_add(d2g_dx2_, tfvar_B, tfvar_B, d2ceval.block(         0, b0 +          0,  1,  1), lambda(cecon_B + j), true);  // tftf
      block_add(d2g_dx2_, tfvar_B, x0var_B, d2ceval.block(         0, b0 +          1,  1, Nx), lambda(cecon_B + j), true);  // tfx0
      block_add(d2g_dx2_, tfvar_B, xfvar_B, d2ceval.block(         0, b0 +     1 + Nx,  1, Nx), lambda(cecon_B + j), true);  // tfxf
      block_add(d2g_dx2_, tfvar_B,  qvar_B, d2ceval.block(         0, b0 + 1 + 2 * Nx,  1, Nq), lambda(cecon_B + j), true);  // tfq

      block_add(d2g_dx2_, x0var_B, x0var_B, d2ceval.block(         1, b0 +          1, Nx, Nx), lambda(cecon_B + j), true);  // x0x0
      block_add(d2g_dx2_, x0var_B, xfvar_B, d2ceval.block(         1, b0 +     1 + Nx, Nx, Nx), lambda(cecon_B + j), true);  // x0xf
      block_add(d2g_dx2_, x0var_B,  qvar_B, d2ceval.block(         1, b0 + 1 + 2 * Nx, Nx, Nq), lambda(cecon_B + j), true);  // x0q

      block_add(d2g_dx2_, xfvar_B, xfvar_B, d2ceval.block(    1 + Nx, b0 +     1 + Nx, Nx, Nx), lambda(cecon_B + j), true);  // xfxf
      block_add(d2g_dx2_, xfvar_B,  qvar_B, d2ceval.block(    1 + Nx, b0 + 1 + 2 * Nx, Nx, Nq), lambda(cecon_B + j), true);  // xfq

      block_add(d2g_dx2_,  qvar_B,  qvar_B, d2ceval.block(1 + 2 * Nx, b0 + 1 + 2 * Nx, Nq, Nq), lambda(cecon_B + j), true);  // qq
      // clang-format on
    }

    d2g_dx2_.makeCompressed();
    return d2g_dx2_;
  }
};

}  // namespace detail
// \endcond

/**
 * @brief Formulate an OCP as a NLP using collocation on a Mesh.
 *
 * @param ocp Optimal control problem definition
 * @param mesh collocation point structure
 * @return encoding of ocp as a nonlinear program
 *
 * @see ocpsol_to_nlpsol(), nlpsol_to_ocpsol()
 */
template<diff::Type DT = diff::Type::Default>
auto ocp_to_nlp(FlatOCPType auto && ocp, MeshType auto && mesh)
  -> detail::OCPNLP<std::decay_t<decltype(ocp)>, std::decay_t<decltype(mesh)>, DT>
{
  return detail::OCPNLP<std::decay_t<decltype(ocp)>, std::decay_t<decltype(mesh)>, DT>(
    std::forward<decltype(ocp)>(ocp), std::forward<decltype(mesh)>(mesh));
}

/**
 * @brief Convert nonlinear program solution to ocp solution
 */
auto nlpsol_to_ocpsol(const FlatOCPType auto & ocp, const MeshType auto & mesh, const NLPSolution & nlp_sol)
{
  using ocp_t = std::decay_t<decltype(ocp)>;

  static constexpr auto Nx  = ocp_t::Nx;
  static constexpr auto Nu  = ocp_t::Nu;
  static constexpr auto Nq  = ocp_t::Nq;
  static constexpr auto Ncr = ocp_t::Ncr;

  const std::size_t N                             = mesh.N_colloc();
  const auto [var_beg, var_len, con_beg, con_len] = detail::ocp_nlp_structure(ocp, mesh);

  const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
  const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

  const auto [dcon_B, qcon_B, crcon_B, cecon_B, m] = con_beg;
  const auto [dcon_L, qcon_L, crcon_L, cecon_L]    = con_len;

  const double t0 = 0;
  const double tf = nlp_sol.x(tfvar_B);

  const Eigen::Vector<double, Nq> Q = nlp_sol.x.segment(qvar_B, qvar_L);

  // state vector has a value at the endpoint

  Eigen::MatrixXd X(ocp.Nx, N + 1);
  X = nlp_sol.x.segment(xvar_B, xvar_L).reshaped(ocp.Nx, xvar_L / ocp.Nx);

  auto xfun = [t0 = t0, tf = tf, mesh = mesh, X = std::move(X)](double t) -> Eigen::Vector<double, Nx> {
    return mesh.template eval<Eigen::Vector<double, Nx>>((t - t0) / (tf - t0), X.colwise(), 0, true);
  };

  // for these we repeat last point since there are no values for endpoint

  Eigen::MatrixXd U(ocp.Nu, N);
  U = nlp_sol.x.segment(uvar_B, uvar_L).reshaped(ocp.Nu, uvar_L / ocp.Nu);

  auto ufun = [t0 = t0, tf = tf, mesh = mesh, U = std::move(U)](double t) -> Eigen::Vector<double, Nu> {
    return mesh.template eval<Eigen::Vector<double, Nu>>((t - t0) / (tf - t0), U.colwise(), 0, false);
  };

  Eigen::MatrixXd Ldyn(ocp.Nx, N);
  Ldyn = nlp_sol.lambda.segment(dcon_B, dcon_L).reshaped(ocp.Nx, dcon_L / ocp.Nx);

  auto ldfun = [t0 = t0, tf = tf, mesh = mesh, Ldyn = std::move(Ldyn)](double t) -> Eigen::Vector<double, Nx> {
    return mesh.template eval<Eigen::Vector<double, Nx>>((t - t0) / (tf - t0), Ldyn.colwise(), 0, false);
  };

  Eigen::MatrixXd Lcr(ocp.Ncr, N);
  Lcr = nlp_sol.lambda.segment(crcon_B, crcon_L).reshaped(ocp.Ncr, crcon_L / ocp.Ncr);

  auto lcrfun = [t0 = t0, tf = tf, mesh = mesh, Lcr = std::move(Lcr)](double t) -> Eigen::Vector<double, Ncr> {
    return mesh.template eval<Eigen::Vector<double, Ncr>>((t - t0) / (tf - t0), Lcr.colwise(), 0, false);
  };

  return OCPSolution<typename ocp_t::X, typename ocp_t::U, ocp_t::Nq, ocp_t::Ncr, ocp_t::Nce>{
    .t0         = t0,
    .tf         = tf,
    .Q          = std::move(Q),
    .u          = std::move(ufun),
    .x          = std::move(xfun),
    .lambda_q   = nlp_sol.lambda.segment(qcon_B, qcon_L),
    .lambda_ce  = nlp_sol.lambda.segment(cecon_B, cecon_L),
    .lambda_dyn = std::move(ldfun),
    .lambda_cr  = std::move(lcrfun),
  };
}

/**
 * @brief Convert ocp solution to nonlinear program solution
 *
 * @note Allocates memory for return type.
 */
NLPSolution ocpsol_to_nlpsol(const FlatOCPType auto & ocp, const MeshType auto & mesh, const auto & ocpsol)
{
  const auto N = mesh.N_colloc();

  const auto [var_beg, var_len, con_beg, con_len] = detail::ocp_nlp_structure(ocp, mesh);

  const auto [tfvar_B, qvar_B, xvar_B, uvar_B, n] = var_beg;
  const auto [tfvar_L, qvar_L, xvar_L, uvar_L]    = var_len;

  const auto [dcon_B, qcon_B, crcon_B, cecon_B, m] = con_beg;
  const auto [dcon_L, qcon_L, crcon_L, cecon_L]    = con_len;

  const double t0 = 0;
  const double tf = ocpsol.tf;

  Eigen::VectorXd x(n), lambda(m);

  x(tfvar_B)                = ocpsol.tf;
  x.segment(qvar_B, qvar_L) = ocpsol.Q;

  lambda.segment(qcon_B, qcon_L)   = ocpsol.lambda_q;
  lambda.segment(cecon_B, cecon_L) = ocpsol.lambda_ce;

  for (const auto & [i, tau] : utils::zip(std::views::iota(0u), mesh.all_nodes())) {
    x.segment(xvar_B + i * ocp.Nx, ocp.Nx) = ocpsol.x(t0 + tau * (tf - t0));
    if (i < N) {
      x.segment(uvar_B + i * ocp.Nu, ocp.Nu)         = ocpsol.u(t0 + tau * (tf - t0));
      lambda.segment(dcon_B + i * ocp.Nx, ocp.Nx)    = ocpsol.lambda_dyn(t0 + tau * (tf - t0));
      lambda.segment(crcon_B + i * ocp.Ncr, ocp.Ncr) = ocpsol.lambda_cr(t0 + tau * (tf - t0));
    }
  }

  return {
    .status = NLPSolution::Status::Unknown,
    .x      = std::move(x),
    .zl     = Eigen::VectorXd::Zero(n),
    .zu     = Eigen::VectorXd::Zero(n),
    .lambda = std::move(lambda),
  };
}

}  // namespace smooth::feedback
