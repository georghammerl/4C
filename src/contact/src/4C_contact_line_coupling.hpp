// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_LINE_COUPLING_HPP
#define FOUR_C_CONTACT_LINE_COUPLING_HPP


#include "4C_config.hpp"

#include "4C_mortar_coupling3d_classes.hpp"

#include <set>

FOUR_C_NAMESPACE_OPEN

namespace Mortar
{
  class ParamsInterface;
}

namespace CONTACT
{
  class Element;

  /*----------------------------------------------------------------------*
   |  LTS / STL coupling                                       farah 07/16|
   *----------------------------------------------------------------------*/
  class LineToSurfaceCoupling3d
  {
   public:
    //! @name Enums and Friends
    enum IntType  // integration types
    {
      lts,  // line to segment
      stl   // segment to line
    };

    /*!
     \brief Constructor with shape function specification

     Constructs an instance of this class and enables custom shape function types.<br>
     Note that this is \b not a collective call as coupling is
     performed in parallel by individual processes.

     */
    LineToSurfaceCoupling3d(Core::FE::Discretization& idiscret, int dim,
        Teuchos::ParameterList& params, Element& pEle, std::shared_ptr<Mortar::Element>& lEle,
        std::vector<Element*> surfEles, LineToSurfaceCoupling3d::IntType itype);

    /*!
     \brief Destructor

     */
    virtual ~LineToSurfaceCoupling3d() = default;
    /*!
     \brief Evaluate coupling (3D)

     */
    void evaluate_coupling();

    //@}
   private:
    /*!
     \brief Build auxiliary plane from target element (3D)

     This method builds an auxiliary plane based on the possibly
     warped source element of this coupling class. This plane is
     defined by the source normal at the source element center.

     */
    virtual bool auxiliary_plane();

    /*!
     \brief Build auxiliary line from source line (3D)

     */
    virtual bool auxiliary_line();

    /*!
     \brief Return center of auxiliary plane

     */
    virtual double* auxc() { return auxc_; }

    /*!
     \brief Return normal of auxiliary line

     */
    virtual double* auxn() { return auxn_; }

    /*!
     \brief Return normal of auxiliary plane

     */
    virtual double* auxn_surf() { return auxn_surf_; }

    /*!
     \brief Get communicator

     */
    virtual MPI_Comm get_comm() const;

    /*!
     \brief create integration lines

     */
    virtual void create_integration_lines(
        std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linvertex);

    /*!
     \brief Get interface discretization

     */
    virtual Core::FE::Discretization& discret() const { return idiscret_; };

    /*!
     \brief Get problem dimension (here: 3D)

     */
    virtual const int& n_dim() { return dim_; };

    /*!
     \brief Get interface contact parameter list

     */
    virtual Teuchos::ParameterList& interface_params() { return imortar_; };

    /*!
     \brief create intersections

     */
    virtual void line_clipping();

    /*!
     \brief create intersections

     */
    virtual bool line_to_line_clipping(Mortar::Vertex& edgeVertex1, Mortar::Vertex& edgeVertex0,
        Mortar::Vertex& lineVertex1, Mortar::Vertex& lineVertex0);

    /*!
     \brief check if all vertices are along a line

     */
    virtual bool check_line_on_line(Mortar::Vertex& edgeVertex1, Mortar::Vertex& edgeVertex0,
        Mortar::Vertex& lineVertex1, Mortar::Vertex& lineVertex0);

    /*!
     \brief perform linearization

     */
    virtual void linearize_vertices(
        std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& linvertex);

    /*!
     \brief perform linearization of line clip

     */
    virtual void lineclip_vertex_linearization(const Mortar::Vertex& currv,
        std::vector<Core::Gen::Pairedvector<int, double>>& currlin,
        const Mortar::Vertex* source_vertex_1, const Mortar::Vertex* source_vertex_2,
        const Mortar::Vertex* target_vertex_1, const Mortar::Vertex* target_vertex_2,
        std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& lin_source_nodes,
        std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& lin_target_nodes);

    /*!
     \brief Get coupling source element

     */
    virtual Element& parent_element() const { return p_ele_; }

    /*!
     \brief Get coupling source element

     */
    virtual std::shared_ptr<Mortar::Element>& line_element() const { return l_ele_; }

    /*!
     \brief Get coupling target elements

     */
    virtual std::vector<Element*> surface_elements() const { return surf_eles_; }

    /*!
     \brief Get coupling target elements

     */
    virtual Element& surface_element() const
    {
      if (curr_ele_ < 0 or curr_ele_ > ((int)surf_eles_.size() - 1))
        FOUR_C_THROW("currEle invalid!");

      return *surf_eles_[curr_ele_];
    }

    /*!
     \brief Get number of target elements

     */
    virtual int number_surface_elements() const { return (int)surf_eles_.size(); }

    /*!
     \brief Get current target element in loop

     */
    virtual int& curr_ele() { return curr_ele_; }

    /*!
     \brief Return length of Auxn() before normalization

     */
    virtual double& lauxn() { return lauxn_; }

    /*!
     \brief Return vector of (projected) source node vertex objects

     */
    virtual std::vector<Mortar::Vertex>& inter_sections() { return intersections_; }

    /*!
      \brief Return vector of (projected) source node vertex objects

    */
    virtual std::vector<Mortar::Vertex>& temp_inter_sections() { return temp_intersections_; }

    /*!
      \brief Return set which guarantee uniqueness of target lines

    */
    virtual std::set<std::pair<int, int>>& done_before() { return donebefore_; }

    /*!
     \brief Return vector of (projected) source node vertex objects

     */
    virtual std::vector<Mortar::Vertex>& source_vertices() { return svertices_; }

    /*!
     \brief Return vector of projected target node vertex objects

     */
    virtual std::vector<Mortar::Vertex>& target_vertices() { return mvertices_; }

    /*!
     \brief Return vector of integration line

     */
    virtual std::shared_ptr<Mortar::IntCell>& int_line() { return int_cell_; }

    /*!
     \brief perform integration for line to segment contact

     */
    virtual void integrate_line();

    /*!
     \brief initialize internal variables

     */
    virtual void initialize();

    /*!
     \brief calculate proper dual shape functions

     */
    virtual void consist_dual_shape();

    /*!
     \brief check orientation of line and target_elem

     */
    virtual bool check_orientation();

    /*!
     \brief Return the 'DerivAuxn' map (vector) of this coupling pair

     */
    virtual std::vector<Core::Gen::Pairedvector<int, double>>& get_deriv_auxn()
    {
      return derivauxn_;
    }
    /*!
     \brief Return the 'DerivAuxc' map (vector) of this coupling pair

     */
    virtual std::vector<Core::Gen::Pairedvector<int, double>>& get_deriv_auxc()
    {
      return derivauxc_;
    }
    //  /*!
    //   \brief Return the 'DerivAuxnLine' map (vector) of this coupling pair
    //
    //   */
    //  virtual std::vector<Core::Gen::Pairedvector<int, double> >& GetDerivAuxnLine()
    //  {
    //    return derivauxnLine_;
    //  }

    /*!
     \brief perform linearization of target vertices

     */
    void target_vertex_linearization(
        std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& currlin);

    /*!
     \brief perform linearization of source vertices

     */
    void source_vertex_linearization(
        std::vector<std::vector<Core::Gen::Pairedvector<int, double>>>& currlin);

    /*!
     \brief Check / set projection status of source nodes (3D)

     This method checks for all source nodes if they are part of the clip
     polygon (equal to any vertex). If so the HasProj status is set true!

     */
    virtual bool has_proj_status();

    /*!
     \brief Projection of source element onto aux. plane (3D)

     This method projects the nodes of the given source CElement
     onto the auxiliary plane derived before.

     */
    virtual bool project_source();

    /*!
     \brief Projection of target element onto aux. plane (3D)

     This method projects the nodes of the current target CElement
     onto the auxiliary plane derived from the source CElement before.

     */
    virtual bool project_target();

    /*!
     \brief Checks roughly whether the two elements are near (3D)

     This methods computes the distance of all target nodes to the
     source element (auxiliary plane). If they are not close, then
     coupling is stopped for the pair.

     */
    virtual bool check_length();

    /*!
    \brief Return integration type

    */
    virtual LineToSurfaceCoupling3d::IntType& i_type() { return int_type_; }

   private:
    //! don't want = operator and cctor
    LineToSurfaceCoupling3d operator=(const LineToSurfaceCoupling3d& old) = delete;
    LineToSurfaceCoupling3d(const LineToSurfaceCoupling3d& old) = delete;

    Core::FE::Discretization& idiscret_;  //< discretization of the contact interface
    int dim_;                             //< problem dimension (here: 3D)

    Element& p_ele_;                           //< parent element connected to line element
    std::shared_ptr<Mortar::Element>& l_ele_;  //< line element to perform coupling for
    std::vector<Element*> surf_eles_;          //< surface elements to perform coupling for
    int curr_ele_;                             //< number of current element

    Teuchos::ParameterList& imortar_;        //< containing contact input parameters
    double auxc_[3];                         //< center of auxiliary plane
    double auxn_[3];                         //< normal of auxiliary plane
    double lauxn_;                           //< length of interpolated Auxn() before normalization
    double auxn_surf_[3];                    //< normal of auxiliary plane of surface element
    int linsize_;                            //< size of lin entries
    std::vector<Mortar::Vertex> svertices_;  //< source node vertex objects
    std::vector<Mortar::Vertex> mvertices_;  //< target node vertex objects
    std::vector<Mortar::Vertex> intersections_;       //< vertex objects for intline
    std::vector<Mortar::Vertex> temp_intersections_;  //< vertex objects for intline temporary
    std::set<std::pair<int, int>>
        donebefore_;  //< set of target node pairs to guarantee uniqueness of line-line clipping
    std::shared_ptr<Mortar::IntCell> int_cell_;  //< vector of integration lines
    std::vector<Core::Gen::Pairedvector<int, double>>
        derivauxn_;  //< derivatives of auxiliary plane normal
    std::vector<Core::Gen::Pairedvector<int, double>>
        derivauxn_line_;  //< derivatives of auxiliary line normal
    std::vector<Core::Gen::Pairedvector<int, double>>
        derivauxc_;  //< derivatives of auxiliary plane normal

    // integration type:
    LineToSurfaceCoupling3d::IntType int_type_;
  };

  /*----------------------------------------------------------------------*
   |  LTL coupling with point contact                          farah 07/16|
   *----------------------------------------------------------------------*/
  class LineToLineCouplingPoint3d
  {
   public:
    /*!
     \brief Constructor with shape function specification

     Constructs an instance of this class and enables custom shape function types.<br>
     Note that this is \b not a collective call as coupling is
     performed in parallel by individual processes.

     */
    LineToLineCouplingPoint3d(Core::FE::Discretization& idiscret, int dim,
        Teuchos::ParameterList& params, std::shared_ptr<Mortar::Element>& lsele,
        std::shared_ptr<Mortar::Element>& lmele);

    /*!
     \brief Destructor

     */
    virtual ~LineToLineCouplingPoint3d() = default;
    /*!
     \brief Evaluate coupling (3D)

     */
    void evaluate_coupling();

    /*!
     \brief perform line projection

     */
    virtual void line_intersection(double* source_xi, double* target_xi,
        Core::Gen::Pairedvector<int, double>& d_source_xi,
        Core::Gen::Pairedvector<int, double>& d_target_xi);

    /*!
     \brief Checks validity

     */
    virtual bool check_intersection(double* source_xi, double* target_xi);

    /*!
     \brief calculate angle between line elements

     */
    virtual double calc_current_angle(Core::Gen::Pairedvector<int, double>& lineAngle);

    /*!
     \brief Checks parallelity

     */
    virtual bool check_parallelity();

    //@}
   private:
    /*!
     \brief evaluate terms

     */
    virtual void evaluate_terms(double* source_xi, double* target_xi,
        Core::Gen::Pairedvector<int, double>& d_source_xi,
        Core::Gen::Pairedvector<int, double>& d_target_xi);

    /*!
     \brief Get communicator

     */
    virtual MPI_Comm get_comm() const;

    /*!
     \brief Get interface discretization

     */
    virtual Core::FE::Discretization& discret() const { return idiscret_; };

    /*!
     \brief Get problem dimension (here: 3D)

     */
    virtual const int& n_dim() { return dim_; };

    /*!
     \brief Get interface contact parameter list

     */
    virtual Teuchos::ParameterList& interface_params() { return imortar_; };


    /*!
     \brief Get coupling source element

     */
    virtual std::shared_ptr<Mortar::Element>& line_source_element() const { return l_sele_; }

    /*!
     \brief Get coupling target element

     */
    virtual std::shared_ptr<Mortar::Element>& line_target_element() const { return l_mele_; }

   private:
    // don't want = operator and cctor
    LineToLineCouplingPoint3d operator=(const LineToLineCouplingPoint3d& old) = delete;
    LineToLineCouplingPoint3d(const LineToLineCouplingPoint3d& old) = delete;

    Core::FE::Discretization& idiscret_;        //< discretization of the contact interface
    int dim_;                                   //< problem dimension (here: 3D)
    Teuchos::ParameterList& imortar_;           //< containing contact input parameters
    std::shared_ptr<Mortar::Element>& l_sele_;  //< line element to perform coupling for
    std::shared_ptr<Mortar::Element>& l_mele_;  //< line element to perform coupling for
  };

}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
