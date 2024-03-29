#ifndef GRID3_H
#define GRID3_H

#include <igl/colormap.h>
#include <Eigen/Core>
#include "Array3T.h"
#include <igl/voxel_grid.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/signed_distance.h>
#include <igl/parallel_for.h>

// #include <tbb/parallel_for.h>
// #include <tbb/blocked_range.h>

class Grid3 {
public:
	Grid3() {}

	// Simple drop-in replacement constructor grid2d with an added z dimension.
	// This defaults to create_mesh = False, useful for representing 3D grids of
	// other components than density, so that it doesn't take up additional
	// memory for the grid at each timestep.
	Grid3(int res_x, int res_y, int res_z, double dx) {
		Grid3(res_x, res_y, res_z, dx, false);
	}

	Grid3(int res_x, int res_y, int res_z, double dx, bool create_mesh) {
		m_res_x = res_x;
		m_res_y = res_y;
		m_res_z = res_z;
		m_dx = dx;

		m_has_mesh = create_mesh;

		// No need to allocate m_x or bounding box or m_GV (which is huge!)
		// if we don't need a mesh for this Grid.
		if (create_mesh) {
			// The bounding box will always be the same size throughout this sim, so
			// set it once on init.
			m_boundingV = Eigen::MatrixXd(8, 3);
			// Centered at x/z origin, with bottom being at y=0
			// This is only for the mesh to simplify imports into other programs.
			// The simulation and all are done with only positive coordinates, and
			// functions like set_m_x take in 0 -> m_res range values.
			m_boundingV <<
				-m_res_x / 2, 0,       -m_res_z / 2,
				m_res_x / 2,  0,       -m_res_z / 2,
				-m_res_x / 2, m_res_y, -m_res_z / 2,
				-m_res_x / 2, 0,       m_res_z / 2,
				m_res_x / 2,  m_res_y, -m_res_z / 2,
				-m_res_x / 2, m_res_y, m_res_z / 2,
				m_res_x / 2,  0,       m_res_z / 2,
				m_res_x / 2,  m_res_y, m_res_z / 2;

			// Allocate memory where buildMesh will put the point values on each
			// call to buildMesh. Do it once here so we don't have to reallocate new
			// memory every time. This makes each iteration faster by ~10ms (on my
			// machine at least lol).
			// We can construct an Eigen array later pretty-much at no cost from
			// this raw array.
			m_x = Eigen::VectorXd(res_x * res_y * res_z);

			// GV will store the coordinates of each grid vertex, which will be the
			// center of a 3D grid that will be used for the Marching Cubes algorithm.
			m_GV = Eigen::MatrixXd(res_x * res_y * res_z, 3);

			// Get the max and min of the bounding box defined above
			const Eigen::Vector3d Vmax = m_boundingV.colwise().maxCoeff();
			const Eigen::Vector3d Vmin = m_boundingV.colwise().minCoeff();

			// Convenience for function-style access from within the lerp helper.
			const Eigen::RowVector3i res(res_x, res_y, res_z);

			// Helper for the GV initialisation below.
			// Linearly interpolates 0 -> res numbers to Vmin -> Vmax numbers
			// The first argument is the number to scale, the second is the
			// dimension to index into V for (x:0, y:1, or z:2)
			const auto lerp = [&](const int di, const int d)->double {
				return Vmin(d) + (double)di / (double)(res(d) - 1) * (Vmax(d) - Vmin(d));
			};

			// Initialize G_V once. This is just the grid vertices, so it will always
			// stay constant.
			for(int i = 0; i < res(2) * res(1) * res(0); i++) {
				// Convert single index to equivalent triple nested loop indices
				const int zi = i / (res(1) * res(0));
				const int yi = (i % (res(1) * res(0))) / res(0);
				const int xi = i % res(0);

				const double z = lerp(zi, 2);
				const double y = lerp(yi, 1);
				const double x = lerp(xi, 0);

				// We use x, y, z, the interpolated values to Vmin->Vmax numbers,
				// since this is the actual position in world coordinates of those
				// vertices.
				m_GV.row(xi + res(0) * yi + res(1) * zi * res(2)) = Eigen::RowVector3d(x, y, z);
			};
		}
	}

	Eigen::VectorXd& x() { return m_x; }

	const Eigen::VectorXd& x() const { return m_x; }

	void buildMesh() {
		if (!m_has_mesh) return;

		// Call the IGL marching cubes algorithm, which generates vertices and
		// faces based on values of a function at points on a grid. The isovalue
		// argument controls the limit considered to be the surface of the mesh.
		// The result is put in m_V and m_F.
		igl::copyleft::marching_cubes(
			m_x,
			m_GV,
			m_res_x,
			m_res_y,
			m_res_z,
			0,
			m_V,
			m_F);
	}

	void reset() {
		if (!m_has_mesh) return;

		m_x.fill(0.0);
		// m_x.fill(1.0);
	}

	void getMesh(Eigen::MatrixXd& ret_V, Eigen::MatrixXi& ret_F) const {
		if (!m_has_mesh) return;

		ret_V = m_V;
		ret_F = m_F;
	}

	// Add a fluid source at the specified location* within the initialised
	// grid. The location argument passed to this function is linearly interpolated
	// to fit within the complete resolution of the grid.
	// * All parameters passed to this function should be within [0, 1]
	void applySource(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax) {
		if (!m_has_mesh) return;

		for (int z = (int)(zmin * m_res_z); z < (int)(zmax * m_res_z); z++) {
			for (int y = (int)(ymin * m_res_y); y < (int)(ymax * m_res_y); y++) {
				for (int x = (int)(xmin * m_res_x); x < (int)(xmax * m_res_x); x++) {
					// Safety check
					int ix = z * m_res_x * m_res_y + y * m_res_x + x;
					if (ix < 0 || ix >= m_res_x * m_res_y * m_res_z) {
						throw std::invalid_argument("only [0, 1]");
					}
					m_x[ix] = 1.f;
				}
			}
		}
	}

	// Add a particle density to the specified coordinates, where values x, y, z
	// are grid positions in range [0, res_x/y/z].
	void set_m_x(int x, int y, int z) {
		if (!m_has_mesh) return;

		// Safety check
		int ix = z * m_res_x * m_res_y + y * m_res_x + x;
		if (ix < 0 || ix >= m_res_x * m_res_y * m_res_z) {
			std::cout << x << " " << y << " " << z << std::endl;
			throw std::invalid_argument("set_m_x got out of bounds");
		}
		m_x[ix] += 1.f;
	}

	void getColors(Eigen::MatrixXd& C) const {
		if (!m_has_mesh) return;

		// Unlike 2D where the mesh could contain parts that were blue and black,
		// 3D meshes only represent the fluid part. So we can return a single
		// value for the whole thing.
		C.resize(1, 3);
		C.row(0) = Eigen::RowVector3d(0, 0, 255);
		return;
	}

	inline int getResolutionX() const {
		return m_res_x;
	}

	inline int getResolutionY() const {
		return m_res_y;
	}

	inline int getResolutionZ() const {
		return m_res_z;
	}

	inline int getGridSpacing() const {
		return m_dx;
	}

protected:
	bool m_has_mesh;
	int m_res_x, m_res_y, m_res_z;
	double m_dx;

	// !! m_x is one-dimensional. Previously it was a three-dimensional lookup
	// but that was slow when passing to the Marching Cubes algorithm in
	// buildMesh.
	Eigen::VectorXd m_x;
	Eigen::MatrixXd m_V;
	Eigen::MatrixXi m_F;
	Eigen::MatrixXd m_GV;

	Eigen::MatrixXd m_boundingV;
};

#endif // GRID3_H
