#ifndef GRID3_H
#define GRID3_H

#include <igl/colormap.h>
#include <Eigen/Core>
#include "Array3T.h"
#include <igl/voxel_grid.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/signed_distance.h>

class Grid3 {
public:
	Grid3() {}

	Grid3(int res_x, int res_y, int res_z, double dx) {
		m_res_x = res_x;
		m_res_y = res_y;
		m_res_z = res_z;
		m_dx = dx;
		m_x = Array3d(res_x, res_y, res_z);

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
	}

	Array3d& x() { return m_x; }

	const Array3d& x() const { return m_x; }

	void buildMesh() {
		const Eigen::Vector3d Vmax = m_boundingV.colwise().maxCoeff();
		const Eigen::Vector3d Vmin = m_boundingV.colwise().minCoeff();
		const Eigen::RowVector3i res = Eigen::RowVector3i(m_res_x, m_res_y, m_res_z);

		Eigen::MatrixXd GV(res(0) * res(1) * res(2), 3);
		Eigen::VectorXd points(res(0) * res(1) * res(2));

		for (int zi = 0; zi < res(2); zi++) {
			// Convert 0 -> res numbers to Vmin -> Vmax numbers
			// The first argument is the number to scale, the second is the
			// dimension to index into V for (x:0, y:1, or z:2)
			const auto lerp = [&](const int di, const int d)->double {
				return Vmin(d) + (double)di / (double)(res(d) - 1) * (Vmax(d) - Vmin(d));
			};
			for (int yi = 0; yi < res(1);yi++) {
				for (int xi = 0; xi < res(0);xi++) {
					const double x = lerp(xi, 0);
					const double y = lerp(yi, 1);
					const double z = lerp(zi, 2);
					// We use x, y, z transformed to Vmin->Vmax numbers, since
					// this is the actual position in world coordinates of
					// those vertices.
					GV.row(xi + res(0) * yi + res(0) * res(1) * zi) = Eigen::RowVector3d(x, y, z);

					// We use xi, yi, and zi coords for m_x since it is 0 -> m_res
					points(xi + res(0) * yi + res(0) * res(1) * zi) = m_x(xi, yi, zi);
				}
			}
		}

		igl::copyleft::marching_cubes(
			points,
			GV,
			res(0),
			res(1),
			res(2),
			0,
			m_V,
			m_F);
	}

	void reset() {
		m_x.zero();
		// m_x.fill(1.0);
	}

	void getMesh(Eigen::MatrixXd& ret_V, Eigen::MatrixXi& ret_F) const {
		ret_V = m_V;
		ret_F = m_F;
	}

	void applySource(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax) {
		for (int z = (int)(zmin * m_res_z); z < (int)(zmax * m_res_z); z++) {
			for (int y = (int)(ymin * m_res_y); y < (int)(ymax * m_res_y); y++) {
				for (int x = (int)(xmin * m_res_x); x < (int)(xmax * m_res_x); x++) {
					m_x(x, y, z) = 1.f;
				}
			}
		}
	}

	void set_m_x(int x, int y, int z) {
		m_x(x, y, z) += 1.f;
	}

	void getColors(Eigen::MatrixXd& C) const {
		// Unlike 2D where the mesh could contain parts that were blue and black,
		// 3D meshes only represent the fluid part. So we can return a single
		// value for the whole thing.
		C.resize(1, 3);
		C.row(0) = Eigen::RowVector3d(0, 255, 255);
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
	Eigen::MatrixXd m_voxel_grid;
	int m_res_x, m_res_y, m_res_z;
	double m_dx;
	Array3d m_x;
	Eigen::MatrixXd m_V;
	Eigen::MatrixXi m_F;

	Eigen::MatrixXd m_boundingV;
};

#endif // GRID3_H
