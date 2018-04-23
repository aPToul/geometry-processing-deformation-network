#include "arap_precompute.h"
#include "laplacian_and_mass.h"
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix_entries.h>
#include <iostream>

using namespace std;

void arap_precompute(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const Eigen::VectorXi & b,
  igl::min_quad_with_fixed_data<double> & data,
  Eigen::SparseMatrix<double> & K,
  int mode)
{
  int num_points = V.rows();
  Eigen::SparseMatrix<double>Laplacian(num_points, num_points);
  Eigen::SparseMatrix<double> Mass(num_points, num_points);
  laplacian_and_mass(V, E, Laplacian, Mass, mode);
  // Update Data with pre computation of the Laplacian
  Eigen::SparseMatrix<double> Aeq;
  igl::min_quad_with_fixed_precompute(Laplacian, b, Aeq, false, data);

  int num_edges = E.rows();
  K.resize(num_points, num_points * 3);

  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(num_edges * 3 * 2);

  // For each edge
  for (int e = 0; e < num_edges; e++) 
  {
    auto edge = E.row(e);
    int V_i = E(0);
    int V_j = E(1);

    Eigen::RowVector3d e_ij;
    if (mode == 0) {
      // Identity edge weight
      e_ij = (V.row(V_i) - V.row(V_j)); 
    }
    if (mode == 1) {
      // c_ij = edge_length_ij, so just normalize
      e_ij = (V.row(V_i) - V.row(V_j)).normalized();
    }

    // For each component Beta of the edge difference vector
    for (int beta = 0; beta < 3; beta++)
    { 
      tripletList.push_back(T(V_i, 3 * V_j + beta, e_ij(beta)));        
      tripletList.push_back(T(V_j, 3 * V_i + beta, -1.0 * e_ij(beta)));
    }
  }
  K.setFromTriplets(tripletList.begin(), tripletList.end());
}
