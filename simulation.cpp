#include <iostream>
#include <fstream>
#include <complex>
#include <random>
#include <omp.h>
typedef std::complex<double> cmplx;
#pragma omp declare reduction(+:cmplx:omp_out+=omp_in)
int main() {
    int ns, nt;
    double m2, epsilon, lambda, tfinal, step, bound;
    std::ifstream fin("input.txt");
    fin >> ns >> nt >> m2 >> epsilon >> lambda >> tfinal >> step >> bound;
    fin.close();
    int *sup = new int[ns];
    int *sdn = new int[ns];
    int *tup = new int[nt];
    int *tdn = new int[nt];
    cmplx *cor = new cmplx[nt];
    cmplx ****phi = new cmplx***[nt];
    cmplx ****psi = new cmplx***[nt];
    double *seed = new double[omp_get_max_threads()];
    double *kmax_thread = new double[omp_get_max_threads()];
    std::mt19937 *rng = new std::mt19937[omp_get_max_threads()];
    std::normal_distribution<> *nor = new std::normal_distribution<>[omp_get_max_threads()];
    for (int i = 0; i < omp_get_max_threads(); ++i) {
        seed[i] = rand();
    }
    double gamma = (double)ns/(double)nt;
    double dt = sqrt(gamma);
    for (int t = 0; t < nt; ++t) {
        if (t == 0) {
            tup[t] = t+1;
            tdn[t] = nt-1;
        }
        else if (t == nt-1) {
            tup[t] = 0;
            tdn[t] = t-1;
        }
        else {
            tup[t] = t+1;
            tdn[t] = t-1;
        }
        phi[t] = new cmplx**[ns];
        psi[t] = new cmplx**[ns];
        for (int x = 0; x < ns; ++x) {
            if (x == 0) {
                sup[x] = x+1;
                sdn[x] = ns-1;
            }
            else if (x == ns-1) {
                sup[x] = 0;
                sdn[x] = x-1;
            }
            else {
                sup[x] = x+1;
                sdn[x] = x-1;
            }
            phi[t][x] = new cmplx*[ns];
            psi[t][x] = new cmplx*[ns];
            for (int y = 0; y < ns; ++y) {
                phi[t][x][y] = new cmplx[ns];
                psi[t][x][y] = new cmplx[ns];
            }
        }
    }
    cmplx **prop = new cmplx*[nt];
    cmplx **invprop = new cmplx*[nt];
    for (int t = 0; t < nt; ++t) {
        prop[t] = new cmplx[2];
        invprop[t] = new cmplx[2];
    }
    for (int n = 0; n < 2; ++n) {
        for (int t = 0; t < nt; ++t) {
            cor[t] = cmplx(0,0);
        }
        for (int i = 0; i < omp_get_max_threads(); ++i) {
            rng[i].seed(seed[i]);
        }
        #pragma omp parallel for
        for (int t = 0; t < nt; ++t) {
            for (int x = 0; x < ns; ++x) {
                for (int y = 0; y < ns; ++y) {
                    for (int z = 0; z < ns; ++z) {
                        phi[t][x][y][z] = sqrt(2*step)*nor[omp_get_thread_num()](rng[omp_get_thread_num()]);
                    }
                }
            }
        }
        int count = 0;
        double time = 0;
        while (time < tfinal) {
            for (int i = 0; i < omp_get_max_threads(); ++i) {
                kmax_thread[i] = 0;
            }
            #pragma omp parallel for
            for (int t = 0; t < nt; ++t) {
                for (int x = 0; x < ns; ++x) {
                    for (int y = 0; y < ns; ++y) {
                        for (int z = 0; z < ns; ++z) {
                            psi[t][x][y][z] =
                            +(phi[tup[t]][x][y][z]-2.0*phi[t][x][y][z]+phi[tdn[t]][x][y][z])/gamma
                            -(phi[t][sup[x]][y][z]-2.0*phi[t][x][y][z]+phi[t][sdn[x]][y][z])*gamma
                            -(phi[t][x][sup[y]][z]-2.0*phi[t][x][y][z]+phi[t][x][sdn[y]][z])*gamma
                            -(phi[t][x][y][sup[z]]-2.0*phi[t][x][y][z]+phi[t][x][y][sdn[z]])*gamma
                            +cmplx(m2,-epsilon)*phi[t][x][y][z]
                            +n*lambda*phi[t][x][y][z]*phi[t][x][y][z]*phi[t][x][y][z];
                            if (abs(psi[t][x][y][z]) > kmax_thread[omp_get_thread_num()]) kmax_thread[omp_get_thread_num()] = abs(psi[t][x][y][z]);
                        }
                    }
                }
            }
            double kmax = 0;
            for (int i = 0; i < omp_get_max_threads(); ++i) {
                if (kmax_thread[i] > kmax) kmax = kmax_thread[i];
            }
            double delta = step*std::min(1.0,bound/kmax);
            #pragma omp parallel for
            for (int t = 0; t < nt; ++t) {
                for (int x = 0; x < ns; ++x) {
                    for (int y = 0; y < ns; ++y) {
                        for (int z = 0; z < ns; ++z) {
                            phi[t][x][y][z] += cmplx(0,-delta)*psi[t][x][y][z]+sqrt(2*delta)*nor[omp_get_thread_num()](rng[omp_get_thread_num()]);
                        }
                    }
                }
            }
            #pragma omp parallel for reduction(+:cor[:nt])
            for (int t = 0; t < nt; ++t) {
                for (int tp = 0; tp < nt; ++tp) {
                    for (int x = 0; x < ns; ++x) {
                        for (int y = 0; y < ns; ++y) {
                            for (int z = 0; z < ns; ++z) {
                                cor[t] += phi[tp][x][y][z]*phi[(t+tp)%nt][x][y][z]/(double)(ns*ns*ns*nt);
                            }
                        }
                    }
                }
            }
            ++count;
            time += delta;
            std::cout << time << '\t' << kmax << '\n';
        }
        for (int k = -nt/2; k < nt/2; ++k) {
            cmplx sum;
            for (int t = 0; t < nt; ++t) {
                sum += cor[t]*exp(cmplx(0,-2*M_PI*k*t/nt))/(double)count;
            }
            prop[k+nt/2][n] = sum;
            invprop[k+nt/2][n] = 1.0/sum;
        }
    }
    std::ofstream fout;
    fout.open("propagator_free.txt");
    for (int k = -nt/2; k < nt/2; ++k) {
        fout << 2*M_PI*k/(nt*dt) << '\t' << real(prop[k+nt/2][0]) << '\t' << imag(prop[k+nt/2][0]) << '\n';
    }
    fout.close();
    fout.open("propagator_interacting.txt");
    for (int k = -nt/2; k < nt/2; ++k) {
        fout << 2*M_PI*k/(nt*dt) << '\t' << real(prop[k+nt/2][1]) << '\t' << imag(prop[k+nt/2][1]) << '\n';
    }
    fout.close();
    fout.open("self_energy.txt");
    for (int k = -nt/2; k < nt/2; ++k) {
        fout << 2*M_PI*k/(nt*dt) << '\t' << real(cmplx(0,1)*(invprop[k+nt/2][0]-invprop[k+nt/2][1])) << '\t' << imag(cmplx(0,1)*(invprop[k+nt/2][0]-invprop[k+nt/2][1])) << '\n';
    }
    fout.close();
    for (int t = 0; t < nt; ++t) {
        delete[] invprop[t];
        delete[] prop[t];
    }
    delete[] invprop;
    delete[] prop;
    for (int t = 0; t < nt; ++t) {
        for (int x = 0; x < ns; ++x) {
            for (int y = 0; y < ns; ++y) {
                delete[] phi[t][x][y];
                delete[] psi[t][x][y];
            }
            delete[] phi[t][x];
            delete[] psi[t][x];
        }
        delete[] phi[t];
        delete[] psi[t];
    }
    delete[] phi;
    delete[] psi;
    delete[] cor;
    delete[] kmax_thread;
    delete[] seed;
    delete[] rng;
    delete[] nor;
    delete[] sup;
    delete[] sdn;
    delete[] tup;
    delete[] tdn;
}
