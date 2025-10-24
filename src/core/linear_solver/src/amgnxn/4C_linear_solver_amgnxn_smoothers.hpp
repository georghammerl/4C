// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_AMGNXN_SMOOTHERS_HPP
#define FOUR_C_LINEAR_SOLVER_AMGNXN_SMOOTHERS_HPP

// Trilinos includes
#include "4C_config.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linear_solver_amgnxn_objects.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_preconditioner_type.hpp"

#include <Epetra_Operator.h>
#include <Ifpack.h>
#include <MueLu.hpp>
#include <MueLu_BaseClass.hpp>
#include <MueLu_Level.hpp>
#include <MueLu_UseDefaultTypes.hpp>
#include <MueLu_Utilities.hpp>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN


namespace Core::LinearSolver::AMGNxN
{
  class GenericSmoother
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~GenericSmoother() = default;

    virtual void solve(
        const BlockedVector& X, BlockedVector& Y, bool InitialGuessIsZero = false) const = 0;

    void richardson(GenericSmoother& Ainv, const BlockedMatrix& A, const BlockedVector& X,
        BlockedVector& Y, int iters, double omega, bool InitialGuessIsZero) const;
    // if InitialGuessIsZero == true we can input any random initial guess and the smoother will
    // take care of making the final result be as if the initial guess would be zero. This
    // avoids to scale to zero the initial guess, and make a little more efficient the smoother
  };

  class SingleFieldSmoother : public GenericSmoother
  {
   public:
    void solve(
        const BlockedVector& X, BlockedVector& Y, bool InitialGuessIsZero = false) const override
    {
      check_single_field_vector(X);
      check_single_field_vector(Y);
      apply(*(X.get_vector(0)), *(Y.get_vector(0)), InitialGuessIsZero);
      return;
    }

    virtual void apply(const Core::LinAlg::MultiVector<double>& X,
        Core::LinAlg::MultiVector<double>& Y, bool InitialGuessIsZero) const = 0;

   protected:
    void check_single_field_vector(const BlockedVector& V) const
    {
      if (not V.has_only_one_block()) FOUR_C_THROW("We need here a single field vector");
      return;
    }
  };

  class BlockedSmoother : public GenericSmoother
  {
  };

  class BgsSmoother : public BlockedSmoother
  {
   public:
    BgsSmoother(Teuchos::RCP<BlockedMatrix> A, std::vector<Teuchos::RCP<GenericSmoother>> smoothers,
        std::vector<std::vector<int>> superblocks, unsigned iter, double omega,
        std::vector<unsigned> iters, std::vector<double> omegas)
        : a_(A),
          smoothers_(smoothers),
          superblocks_(superblocks),
          iter_(iter),
          omega_(omega),
          iters_(iters),
          omegas_(omegas)
    {
    }

    void solve(
        const BlockedVector& X, BlockedVector& Y, bool InitialGuessIsZero = false) const override;

   private:
    Teuchos::RCP<BlockedMatrix> a_;
    std::vector<Teuchos::RCP<GenericSmoother>> smoothers_;
    std::vector<std::vector<int>> superblocks_;
    unsigned iter_;
    double omega_;
    std::vector<unsigned> iters_;
    std::vector<double> omegas_;
  };

  // Forward declarations
  class Hierarchies;
  class VcycleSingle;

  class MueluSmootherWrapper : public SingleFieldSmoother
  {
   public:
    MueluSmootherWrapper(
        Teuchos::RCP<MueLu::SmootherBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>> S)
        : s_(S)
    {
    }

    void apply(const Core::LinAlg::MultiVector<double>& X, Core::LinAlg::MultiVector<double>& Y,
        bool InitialGuessIsZero = false) const override;

   private:
    Teuchos::RCP<MueLu::SmootherBase<Scalar, LocalOrdinal, GlobalOrdinal, Node>> s_;
  };

  class MueluHierarchyWrapper : public SingleFieldSmoother  // Not used
  {
   public:
    MueluHierarchyWrapper(
        Teuchos::RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node>> H);

    void apply(const Core::LinAlg::MultiVector<double>& X, Core::LinAlg::MultiVector<double>& Y,
        bool InitialGuessIsZero = false) const override;

   private:
    Teuchos::RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node>> h_;
    Teuchos::RCP<Epetra_Operator> p_;
  };

  class MueluAMGWrapper : public SingleFieldSmoother
  {
   public:
    MueluAMGWrapper(Teuchos::RCP<Core::LinAlg::SparseMatrix> A, int num_pde, int null_space_dim,
        std::shared_ptr<std::vector<double>> null_space_data,
        const Teuchos::ParameterList& muelu_list);

    void apply(const Core::LinAlg::MultiVector<double>& X, Core::LinAlg::MultiVector<double>& Y,
        bool InitialGuessIsZero = false) const override;

    void setup();

   protected:
    Teuchos::RCP<Core::LinAlg::SparseMatrix> A_;
    int num_pde_;
    int null_space_dim_;
    std::shared_ptr<std::vector<double>> null_space_data_;
    Teuchos::ParameterList muelu_list_;
    Teuchos::RCP<MueLu::Hierarchy<Scalar, LocalOrdinal, GlobalOrdinal, Node>> H_;
    void build_hierarchy();

   private:
    Teuchos::RCP<Epetra_Operator> p_;
  };

  class SingleFieldAMG : public MueluAMGWrapper
  {
   public:
    SingleFieldAMG(Teuchos::RCP<Core::LinAlg::SparseMatrix> A, int num_pde, int null_space_dim,
        std::shared_ptr<std::vector<double>> null_space_data,
        const Teuchos::ParameterList& muelu_list, const Teuchos::ParameterList& fine_smoother_list);

    void apply(const Core::LinAlg::MultiVector<double>& X, Core::LinAlg::MultiVector<double>& Y,
        bool InitialGuessIsZero = false) const override;

   private:
    Teuchos::ParameterList fine_smoother_list_;
    Teuchos::RCP<VcycleSingle> v_;
    void setup();
  };

  // Auxiliary class to wrap the null space data to be used within the smoothers
  class NullSpaceInfo
  {
   public:
    NullSpaceInfo() {}
    NullSpaceInfo(
        int num_pdes, int null_space_dim, std::shared_ptr<std::vector<double>> null_space_data)
        : num_pdes_(num_pdes), null_space_dim_(null_space_dim), null_space_data_(null_space_data)
    {
    }

    int get_num_pd_es() { return num_pdes_; }
    int get_null_space_dim() { return null_space_dim_; }
    std::shared_ptr<std::vector<double>> get_null_space_data() { return null_space_data_; }

   private:
    int num_pdes_;
    int null_space_dim_;
    std::shared_ptr<std::vector<double>> null_space_data_;
  };

  class Hierarchies;  // forward declaration

  class SmootherManager  // TODO: this is quite lengthy. This can be done with a ParameterList
  {
   public:
    SmootherManager();
    Teuchos::RCP<BlockedMatrix> get_operator();
    Teuchos::ParameterList get_params();
    Teuchos::ParameterList get_params_smoother();
    Teuchos::RCP<Hierarchies> get_hierarchies();
    int get_level();
    int get_block();
    std::vector<int> get_blocks();
    std::string get_smoother_name();
    std::string get_type();
    std::string get_verbosity();
    NullSpaceInfo get_null_space();
    std::vector<NullSpaceInfo> get_null_space_all_blocks();

    void set_operator(Teuchos::RCP<BlockedMatrix> in);
    void set_params(const Teuchos::ParameterList& in);
    void set_params_smoother(const Teuchos::ParameterList& in);
    void set_hierarchies(Teuchos::RCP<Hierarchies> in);
    void set_level(int in);
    void set_block(int in);
    void set_blocks(std::vector<int> in);
    void set_smoother_name(std::string in);
    void set_type(std::string in);
    void set_verbosity(std::string in);
    void set_null_space(const NullSpaceInfo& in);
    void set_null_space_all_blocks(const std::vector<NullSpaceInfo>& in);

    bool is_set_operator();
    bool is_set_params();
    bool is_set_params_smoother();
    bool is_set_hierarchies();
    bool is_set_level();
    bool is_set_block();
    bool is_set_blocks();
    bool is_set_smoother_name();
    bool is_set_type();
    bool is_set_verbosity();
    bool is_set_null_space();
    bool is_set_null_space_all_blocks();

   private:
    Teuchos::RCP<BlockedMatrix> operator_;
    Teuchos::ParameterList params_;
    Teuchos::ParameterList params_subsolver_;
    Teuchos::RCP<Hierarchies> hierarchies_;
    int level_;
    int block_;
    std::vector<int> blocks_;
    std::string subsolver_name_;
    std::string type_;
    std::string verbosity_;
    NullSpaceInfo null_space_;
    std::vector<NullSpaceInfo> null_space_all_blocks_;

    bool set_operator_;
    bool set_params_;
    bool set_params_subsolver_;
    bool set_hierarchies_;
    bool set_level_;
    bool set_block_;
    bool set_blocks_;
    bool set_subsolver_name_;
    bool set_type_;
    bool set_verbosity_;
    bool set_null_space_;
    bool set_null_space_all_blocks_;
  };

  class SmootherFactoryBase : public SmootherManager
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~SmootherFactoryBase() = default;

    virtual Teuchos::RCP<GenericSmoother> create() = 0;
  };

  // This class is able to create any smoother. The smoother to be created is given in a
  // parameter list
  class SmootherFactory : public SmootherFactoryBase
  {
   public:
    Teuchos::RCP<GenericSmoother> create() override;

   private:
    void set_type_and_params();
  };

  class BgsSmootherFactory : public SmootherFactoryBase
  {
   public:
    Teuchos::RCP<GenericSmoother> create() override;

   private:
    void parse_smoother_names(const std::string& smoothers_string,
        std::vector<std::string>& smoothers_vector, std::vector<std::vector<int>> superblocks);
  };

  class MueluSmootherWrapperFactory : public SmootherFactoryBase
  {
   public:
    Teuchos::RCP<GenericSmoother> create() override;
  };

  class MueluAMGWrapperFactory : public SmootherFactoryBase
  {
   public:
    Teuchos::RCP<GenericSmoother> create() override;
  };

}  // namespace Core::LinearSolver::AMGNxN

FOUR_C_NAMESPACE_CLOSE

#endif
