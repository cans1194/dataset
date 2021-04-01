import numpy as np
import pickle
import matplotlib.pyplot as plt


def calc_vec_energy(vec):
    i_squared = np.power(vec[0], 2.0)
    q_squared = np.power(vec[1], 2.0)
    inst_energy = np.sqrt(i_squared + q_squared)
    return sum(inst_energy)


def calc_mod_energies(ds):
    for modulation, snr in ds:
        avg_energy = 0
        n_vectors = ds[(modulation, snr)].shape[0]
        for vec in ds[(modulation, snr)]:
            avg_energy += calc_vec_energy(vec)
        avg_energy /= n_vectors
        print(f"{modulation} at {snr} has {n_vectors} vectors avg energy of {avg_energy}")


def calc_mod_bias(ds):
    for modulation, snr in ds:
        avg_bias_re = 0
        avg_bias_im = 0
        n_vectors = ds[(modulation, snr)].shape[0]
        for vec in ds[(modulation, snr)]:
            avg_bias_re += (np.mean(vec[0]))
            avg_bias_im += (np.mean(vec[1]))
        # avg_bias_re /= n_vectors
        # avg_bias_im /= n_vectors
        print(f"{modulation} at {snr} has {n_vectors} vectors avg bias of {avg_bias_re} + {avg_bias_im} j")


def calc_mod_stddev(ds):
    for modulation, snr in ds:
        avg_stddev = 0
        n_vectors = ds[(modulation, snr)].shape[0]
        for vec in ds[(modulation, snr)]:
            avg_stddev += np.abs(np.std(vec[0]+1j*vec[1]))
        # avg_stddev /= n_vectors
        print(f"{modulation} at {snr} has {n_vectors} vectors avg stddev of {avg_stddev}")


def open_ds(location="X_4_dict.dat"):
    f = open(location)
    ds = pickle.load(f)
    return ds


def main():
    ds = open_ds()
    # plt.plot(ds[('BPSK', 12)][25][0][:])
    # plt.plot(ds[('BPSK', 12)][25][1][:])
    # plt.show()
    # calc_mod_energies(ds)
    # calc_mod_stddev(ds)
    calc_mod_bias(ds)


if __name__ == "__main__":
    main()
