import numpy as np
import pickle


def generate_modal_mask(modal_MR, lengths, uids):
    assert all(
        0 <= mr < 1 for mr in modal_MR), "Modal missing rates should be between 0 and 1."

    num_sample = sum(lengths)
    count = 0
    mask = []

    # atv
    ttt = int(num_sample*(1-modal_MR[0])*(1-modal_MR[1])*(1-modal_MR[2]))
    mask.extend([[True, True, True]] * (ttt))
    count += ttt

    # at
    ttf = int(num_sample*(1-modal_MR[0])*(1-modal_MR[1])*modal_MR[2])
    mask.extend([[True, True, False]] * ttf)
    count += ttf

    # tv
    ftt = int(num_sample*modal_MR[0]*(1-modal_MR[1])*(1-modal_MR[2]))
    mask.extend([[False, True, True]] * ftt)
    count += ftt

    # av
    tft = int(num_sample*(1-modal_MR[0])*modal_MR[1]*(1-modal_MR[2]))
    mask.extend([[True, False, True]] * tft)
    count += tft

    # a
    tff = int(num_sample*(1-modal_MR[0])*modal_MR[1]*modal_MR[2])
    mask.extend([[True, False, False]] * tff)
    count += tff

    # t
    ftf = int(num_sample*modal_MR[0]*(1-modal_MR[1])*modal_MR[2])
    mask.extend([[False, True, False]] * ftf)
    count += ftf

    # v
    fft = int(num_sample*modal_MR[0]*modal_MR[1]*(1-modal_MR[2]))
    mask.extend([[False, False, True]] * fft)
    count += fft

    # no modal
    left = num_sample - count
    tmp = left // 3
    mask.extend([[True, False, False]] * tmp)
    mask.extend([[False, False, True]] * tmp)
    mask.extend([[False, True, False]] * (left - 2 * tmp))
    
    modal_mask = {"a": {}, "t": {}, "v": {}}
        
    count = 0
    for length, uid in zip(lengths, uids):
        sample_mask = np.array(mask[count:count + length])

        modal_mask["a"][uid] = sample_mask[:, 0]
        modal_mask["t"][uid] = sample_mask[:, 1]
        modal_mask["v"][uid] = sample_mask[:, 2]

        count += length

    return modal_mask
