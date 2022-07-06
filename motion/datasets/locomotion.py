import os
import numpy as np
from .motion_data import MotionDataset, TestDataset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')

def mirror_data(data):
    aa = data.copy()
    aa[:,:,3:15]=data[:,:,15:27]
    aa[:,:,3:15:3]=-data[:,:,15:27:3]
    aa[:,:,15:27]=data[:,:,3:15]
    aa[:,:,15:27:3]=-data[:,:,3:15:3]
    aa[:,:,39:51]=data[:,:,51:63]
    aa[:,:,39:51:3]=-data[:,:,51:63:3]
    aa[:,:,51:63]=data[:,:,39:51]
    aa[:,:,51:63:3]=-data[:,:,39:51:3]
    aa[:,:,63]=-data[:,:,63]
    aa[:,:,65]=-data[:,:,65]
    return aa

def reverse_time(data):
    aa = data[:,-1::-1,:].copy()
    aa[:,:,63] = -aa[:,:,63]
    aa[:,:,64] = -aa[:,:,64]
    aa[:,:,65] = -aa[:,:,65]
    return aa

def inv_standardize(data, scaler):
    shape = data.shape
    flat = data.reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled

def fit_and_standardize(data):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler

def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled

def create_synth_test_data(n_frames, nFeats, scaler):

    syth_data = np.zeros((7,n_frames,nFeats))
    lo_vel = 1.0
    hi_vel = 2.5
    lo_r_vel = 0.08
    hi_r_vel = 0.08
    syth_data[0,:,63:65] = 0
    syth_data[1,:,63] = lo_vel
    syth_data[2,:,64] = lo_vel
    syth_data[3,:,64] = hi_vel
    syth_data[4,:,64] = -lo_vel
    syth_data[5,:,64] = lo_vel
    syth_data[5,:,65] = lo_r_vel
    syth_data[6,:,64] = hi_vel
    syth_data[6,:,65] = hi_r_vel
    syth_data = standardize(syth_data, scaler)
    syth_data[:,:,:63] = np.zeros((syth_data.shape[0],syth_data.shape[1],63))
    return syth_data.astype(np.float32)

class Locomotion():

    def __init__(self, hparams, is_training, log_path):

        data_root = hparams.Dir.data_root
        self.is_training = is_training
        if is_training:
            train_series = np.load(os.path.join(data_root, 'all_multi_context.npz'), allow_pickle=True)[
                'train'].item()
            train_data = train_series["motion"]
            train_style = train_series["style"]
            train_data = np.array(train_data).astype(np.float32)
        
        test_series = np.load(os.path.join(data_root, 'all_multi_context.npz'), allow_pickle=True)[
            'test'].item()
        test_data = test_series["motion"]
        test_style = test_series["style"]
        test_data = np.array(test_data).astype(np.float32)

        FGD_val_series = np.load(os.path.join(data_root, 'all_multi_context.npz'), allow_pickle=True)[
            'val'].item()
        FGD_val_data = FGD_val_series["motion"]
        FGD_val_style = FGD_val_series["style"]
        FGD_val_data = np.array(FGD_val_data).astype(np.float32)

        print("test_data: " + str(test_data.shape))
        print("FGD_val_data: " + str(FGD_val_data.shape))

        if is_training:
            print("input_data: " + str(train_data.shape))

            # Split into train and val sets
            validation_data = train_data[-1000:, :, :]
            validation_style = train_style[-1000:]
            train_data = train_data[:-1000, :, :]
            train_style = train_style[:-1000]   

        # Data augmentation
        if is_training and hparams.Data.mirror:
            mirrored = mirror_data(train_data)
            train_data = np.concatenate((train_data, mirrored), axis=0)

        if is_training and hparams.Data.reverse_time:
            rev = reverse_time(train_data)
            train_data = np.concatenate((train_data, rev), axis=0)

        audio_start = 29 
        if is_training:
            # Standardize
            train_motion_data = train_data[:, :, audio_start:]
            train_audio_data = train_data[:, :, :audio_start]

            val_motion_data = validation_data[:, :, audio_start:]
            val_audio_data = validation_data[:, :, :audio_start]

        test_motion_data = test_data[:, :, audio_start:]
        test_audio_data = test_data[:, :, :audio_start]

        FGD_motion_data = FGD_val_data[:, :, audio_start:]
        FGD_audio_data = FGD_val_data[:, :, :audio_start]

        data_state_dict = {}
        if is_training:
            train_motion_data, m_scaler = fit_and_standardize(train_motion_data)
            val_motion_data = standardize(val_motion_data, m_scaler)
            data_state_dict["m_scaler"] = m_scaler
        else:
            m_scaler = np.load(os.path.join(log_path, "data_state_dict.npy"), allow_pickle=True).item()["m_scaler"]
        
        test_motion_data = standardize(test_motion_data, m_scaler)
        FGD_motion_data = standardize(FGD_motion_data, m_scaler)

        if is_training:
            train_audio_data, a_scaler = fit_and_standardize(train_audio_data)
            data_state_dict["a_scaler"] = a_scaler
            val_audio_data = standardize(val_audio_data, a_scaler)
        else:
            a_scaler = np.load(os.path.join(log_path, "data_state_dict.npy"), allow_pickle=True).item()["a_scaler"]
        
        test_audio_data = standardize(test_audio_data, a_scaler)
        FGD_audio_data = standardize(FGD_audio_data, a_scaler)

        if is_training:
            np.save(os.path.join(log_path, "data_state_dict.npy"), data_state_dict)

        if is_training:
            train_data = np.concatenate((train_audio_data, train_motion_data), axis=2)
            validation_data = np.concatenate((val_audio_data, val_motion_data), axis=2)
        
        test_data = np.concatenate((test_audio_data, test_motion_data), axis=2)
        FGD_data = np.concatenate((FGD_audio_data, FGD_motion_data), axis=2)

        all_test_data = test_data
        all_FGD_data = FGD_data

        
        self.n_test = all_test_data.shape[0]
        n_tiles = 1 + hparams.Train.batch_size // self.n_test
        self.n_FGD = all_FGD_data.shape[0]
        n_tiles_FGD =  1 + hparams.Train.batch_size // self.n_FGD

        all_test_data = np.tile(all_test_data.copy(), (n_tiles, 1, 1))
        all_FGD_data =  np.tile(all_FGD_data.copy(), (n_tiles_FGD, 1, 1))

        self.scaler = m_scaler
        self.a_scaler = a_scaler
        self.frame_rate = hparams.Data.framerate

        # Create pytorch data sets
        if is_training:
            self.train_dataset = MotionDataset(train_data[:, :, :audio_start], train_data[:, :, audio_start:], train_style,
                                           hparams.Data.seqlen,
                                           hparams.Data.n_lookahead, hparams.Data.dropout)
            self.validation_dataset = MotionDataset(validation_data[:, :, :audio_start], validation_data[:, :, audio_start:], validation_style,
                                                hparams.Data.seqlen, hparams.Data.n_lookahead, hparams.Data.dropout)
        
        self.test_dataset = TestDataset(all_test_data[:, :, :audio_start], all_test_data[:, :, audio_start:], test_style)
        self.FGD_val_dataset = TestDataset(all_FGD_data[:, :, :audio_start], all_FGD_data[:, :, audio_start:], FGD_val_style)
        self.seqlen = hparams.Data.seqlen

        self.n_x_channels = (all_test_data.shape[2] - audio_start) #* clip_length
        self.n_cond_channels = self.n_x_channels * (hparams.Data.seqlen) + audio_start * (
                hparams.Data.seqlen + 1 + hparams.Data.n_lookahead)
        
        if is_training:
            self.control_shape = self.train_dataset.get_control_shape()
            self.autoreg_shape = self.train_dataset.get_autoreg_shape()
        
        print("prepare data.")

    def n_channels(self):
        return self.n_x_channels, self.n_cond_channels
    
    def get_control_shape(self):
        return self.control_shape[2] # [feature]
    
    def get_autoreg_shape(self):
        return self.autoreg_shape[2] # [feature]

    def get_train_dataset(self):
        if self.is_training:
            return self.train_dataset
        else:
            return None

    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset
    
    def get_FGD_val_dataset(self):
        return self.FGD_val_dataset

