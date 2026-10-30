"""Train the FaultGuard AI fault-type model on simulated faults and test it on separate simulated records.

    python train_faultguard_model.py

Every training record goes through the same pipeline as a real record (waveforms with DC offset, noise and breaker
opening -> fault detection -> one-cycle Fourier phasors -> features), so the model sees what it will see in the field."""
import pickle
import faultguard_core as fg

if __name__ == '__main__':
    b = fg.train_model(6000, 1500)
    with open('faultguard_model.pkl', 'wb') as f:
        pickle.dump(b, f)
    print('rule', round(b['rule_accuracy']*100,1), 'highRf', b['acc_high_rf'], b['n_high_rf'], 'CTsat', b['acc_ct_sat'], b['n_ct_sat'], 'loc lowRf', b['loc_err_lowrf'])
    print('accuracy', round(b['accuracy'] * 100, 2), '% on', b['n_test'], 'test records; location median error', b['loc_err_all'], '% of line, p90', b['loc_err_p90'])
    print(b['loc_err'])
    import numpy as np
    print(np.array(b['confusion']))
