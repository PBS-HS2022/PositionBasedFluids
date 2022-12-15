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
		buildMesh();
	}

	Array3d& x() { return m_x; }

	const Array3d& x() const { return m_x; }

	void buildMesh() {
		Eigen::MatrixXi F(12, 3);
		Eigen::MatrixXd V(8, 3);
		V << 0, 0, 0,
			 m_res_x, 0, 0,
			 0, m_res_y, 0,
			 0, 0, m_res_z,
			 m_res_x, m_res_y, 0,
			 0, m_res_y, m_res_z,
			 m_res_x, 0, m_res_z,
			 m_res_x, m_res_y, m_res_z;
		F << 0, 1, 4,
		     0, 4, 2,
			 1, 6, 7,
			 1, 7, 4,
			 6, 3, 5,
			 6, 5, 7,
			 3, 0, 2,
			 3, 2, 5,
			 2, 4, 7,
			 2, 7, 5,
			 3, 6, 1,
			 3, 1, 0;

		Eigen::Vector3d Vmax(m_res_x - 1, m_res_y - 1, m_res_z - 1);
		Eigen::Vector3d Vmin(0.f, 0.f, 0.f);
		const Eigen::RowVector3i res = (Vmax-Vmin).cast<int>();
		// create grid
		// std::cout<<"Creating grid..."<<std::endl;
		Eigen::MatrixXd GV(res(0)*res(1)*res(2), 3);
		Eigen::VectorXd points(res(0)*res(1)*res(2));

		for (int zi = 0;zi<res(2);zi++) {
			const auto lerp = [&](const int di, const int d)->double {
				return Vmin(d)+(double)di/(double)(res(d)-1)*(Vmax(d)-Vmin(d));
			};
			const double z = lerp(zi,2);
			for (int yi = 0;yi<res(1);yi++) {
				const double y = lerp(yi,1);
				for (int xi = 0;xi<res(0);xi++) {
					const double x = lerp(xi,0);
					GV.row(xi+res(0)*(yi + res(1)*zi)) = Eigen::RowVector3d(x,y,z);
					points(xi + res(0) * (yi + res(1) * zi)) = m_x(x, y, z);
				}
			}
		}

		igl::copyleft::marching_cubes(points, GV, res(0), res(1), res(2), 0, m_V, m_F);
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
					// m_x(x, y) = 1.0;
					m_x(x, y, z) = 0.5;
				}
			}
		}
	}

	void set_m_x(int x, int y, int z) {
		// std::cout << m_res_x << " " << m_res_y << std::endl;
		m_x(x, y, z) = 0.5;
	}

	void getColors(Eigen::MatrixXd& C, bool normalize=false, bool faceColor=true) const { 
		C.resize(1, 3);
		C.row(0) = Eigen::RowVector3d(0, 0, 255);
		return;
		if (faceColor) {
			if (C.rows() == 0) {
				int num_faces = m_res_x * m_res_y * m_res_z * 2;
				C = Eigen::MatrixXd(num_faces, 3);
			}
			int i = 0;
			double cmin = m_x(0, 0, 0);
			double cmax = cmin;
			for (int z = 0; z < m_res_z; ++z) {
				for (int y = 0; y < m_res_y; ++y) {
					for (int x = 0; x < m_res_x; ++x) {
						double c = m_x(x, y, z);
						if (normalize) {
							if (c > cmax) cmax = c;
							if (c < cmin) cmin = c;
						}
						else {
							for (int f = 0; f < 2; f++) {
								// setting each of the 12 faces
								if (c == 0.5) {
									C.row(i++) = Eigen::RowVector3d(0, 0, 255);
								}
								else {
									C.row(i++).setConstant(c);
								}
							}
						}
					}
				}
			}

			if (!normalize) return;
			else if (cmin == cmax) {
				C.setZero();
				return;
			}

			// std::cout << "cmin:" << cmin << " cmax:" << cmax << std::endl;
			for (int z = 0; z < m_res_z; ++z) {
				for (int y = 0; y < m_res_y; ++y) {
					for (int x = 0; x < m_res_x; ++x) {
						double c = m_x(x, y, 0);

						c = (c - cmin) / (cmax - cmin); // [0,1]
						double r, g, b;
						igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, c, r, g, b);
						for (int f = 0; f < 2; f++) {
							C.row(i++) = Eigen::RowVector3d(r, g, b);
						}
					}
				}
			}
		}
		else {
			// vertex color
			if (C.rows() == 0) {
				int num_vertices = (m_res_x + 1) * (m_res_y + 1) * (m_res_z + 1);
				C = Eigen::MatrixXd(num_vertices, 3);
			}
			int i = 0;
			double cmin = m_x(0, 0, 0);
			double cmax = cmin;
			for (int y = 0; y <= m_res_y; ++y) {
				for (int x = 0; x <= m_res_x; ++x) {
					int x0 = std::max(x - 1, 0);
					int x1 = std::min(x, m_res_x - 1);
					int y0 = std::max(y - 1, 0);
					int y1 = std::min(y, m_res_y - 1);

					double c00 = m_x(x0, y0, 0);
					double c01 = m_x(x0, y1, 0);
					double c10 = m_x(x1, y0, 0);
					double c11 = m_x(x1, y1, 0);
					double c = (c00 + c01 + c10 + c11) / 4;
					if (normalize) {
						if (c > cmax) cmax = c;
						if (c < cmin) cmin = c;
					}
					C.row(i++).setConstant(c);
				}
			}

			if (!normalize) return;
			else if (cmin == cmax) {
				C.setZero();
				return;
			}

			i = 0;
			// std::cout << "cmin:" << cmin << " cmax:" << cmax << std::endl;
			for (int y = 0; y <= m_res_y; ++y) {
				for (int x = 0; x <= m_res_x; ++x) {
					double c = (C(i, 0) - cmin) / (cmax - cmin); // [0,1]
					double r, g, b;
					igl::colormap(igl::COLOR_MAP_TYPE_VIRIDIS, c, r, g, b);
					C.row(i++) = Eigen::RowVector3d(r, g, b);
				}
			}
		}
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
};

#endif // GRID3_H
