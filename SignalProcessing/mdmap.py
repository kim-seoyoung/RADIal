import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift, fft2
import os

class MicroDopplerGenerator:
    """
    RADIal 데이터셋용 마이크로 도플러 스펙트로그램 생성기
    시간에 따른 도플러 시그니처를 분석하여 미세 운동 감지
    """
    
    def __init__(self, calibration_table_path):
        """
        Args:
            calibration_table_path: CalibrationTable.npy 경로
        """
        self.calib_table = np.load(calibration_table_path)
        
        # RADIal 레이더 사양 [web:29][web:57]
        self.num_rx = 16  # 수신 안테나
        self.num_tx = 12  # 송신 안테나
        self.num_virtual = 192  # 가상 안테나 (16*12)
        
        # ADC 데이터 파라미터
        self.num_samples_per_chirp = 512  # Range samples
        self.num_chirps_per_frame = 256    # Chirps per frame
        self.num_rx_per_chip = 4          # 칩당 Rx 안테나 개수
        
        # FMCW 파라미터 (실제 RADIal 파라미터로 조정 필요)
        self.fs = 5e6          # ADC 샘플링 주파수 (Hz)
        self.fc = 77e9         # 중심 주파수 (77 GHz)
        self.bandwidth = 400e6 # 대역폭 (Hz)
        self.chirp_time = 50e-6  # Chirp 시간 (초)
        self.frame_rate = 20   # 프레임 레이트 (Hz)
        
        self.c = 3e8  # 빛의 속도
        self.wavelength = self.c / self.fc
        
    def load_adc_data(self, sequence_path, frame_idx):
        """
        RADIal ADC 데이터 로드
        
        Args:
            sequence_path: 시퀀스 폴더 경로
            frame_idx: 프레임 인덱스
            
        Returns:
            adc_data: [num_chirps, num_rx, num_samples] 형태의 복소수 배열
        """
        adc_data = np.zeros((self.num_chirps_per_frame, self.num_rx, 
                            self.num_samples_per_chirp), dtype=np.complex64)
        
        # Frame size calculations
        # One frame per chip: num_samples * num_rx_per_chip * num_chirps
        frame_size_complex = self.num_samples_per_chirp * self.num_rx_per_chip * self.num_chirps_per_frame
        # 4 bytes per complex sample (2 bytes I + 2 bytes Q)
        frame_size_bytes = frame_size_complex * 4
        
        seq_name = os.path.basename(sequence_path.rstrip(os.sep))
        
        # 4개의 칩 파일 읽기 [web:29]
        for chip_id in range(4):
            # Try original filename pattern
            adc_file = os.path.join(sequence_path, f'RECORD@{frame_idx:06d}@RADAR_{chip_id}.bin')
            if not os.path.exists(adc_file):
                # Try standard filename pattern e.g. RECORD@Timestamp_radar_ch0.bin
                adc_file = os.path.join(sequence_path, f'{seq_name}_radar_ch{chip_id}.bin')
            
            if os.path.exists(adc_file):
                try:
                    # Calculate offset
                    offset = frame_idx * frame_size_bytes
                    
                    # Read specific frame using memmap
                    # We read 2 * frame_size_complex (int16) elements
                    raw_data = np.memmap(adc_file, dtype=np.int16, mode='r', 
                                         offset=offset, 
                                         shape=(frame_size_complex * 2,))
                    
                    # I/Q 데이터 분리 및 복소수 변환
                    I = raw_data[0::2].astype(np.float32)
                    Q = raw_data[1::2].astype(np.float32)
                    complex_data = I + 1j * Q
                    
                    # 형태 재구성: 
                    # rpl.py: reshape to (Samples, Rx, Chirps) with order='F'
                    complex_data = complex_data.reshape(
                        self.num_samples_per_chirp,
                        self.num_rx_per_chip,
                        self.num_chirps_per_frame,
                        order='F'
                    )
                    
                    # mdmap.py expects [chirps, rx, samples]
                    # Transpose (Samples, Rx, Chirps) -> (Chirps, Rx, Samples)
                    # Indices: (0, 1, 2) -> (2, 1, 0)
                    complex_data = complex_data.transpose(2, 1, 0)
                    
                    # 전체 배열에 삽입
                    rx_start = chip_id * self.num_rx_per_chip
                    rx_end = rx_start + self.num_rx_per_chip
                    adc_data[:, rx_start:rx_end, :] = complex_data
                    
                except Exception as e:
                    print(f"Error reading chip {chip_id}: {e}")
                    pass
                
        return adc_data
    
    def compute_range_doppler(self, adc_data, apply_calib=True):
        """
        Range-Doppler 맵 생성 [web:42]
        
        Args:
            adc_data: [num_chirps, num_rx, num_samples]
            apply_calib: 캘리브레이션 테이블 적용 여부
            
        Returns:
            rd_map: [num_samples, num_chirps, num_rx] 형태의 Range-Doppler 맵
        """
        # Range FFT (fast-time)
        window_range = np.hanning(self.num_samples_per_chirp)
        adc_windowed = adc_data * window_range[np.newaxis, np.newaxis, :]
        range_fft = fft(adc_windowed, axis=2)
        
        # 캘리브레이션 적용
        if apply_calib and self.calib_table is not None:
            range_fft = range_fft * self.calib_table[np.newaxis, :, np.newaxis]
        
        # Doppler FFT (slow-time)
        window_doppler = np.hanning(self.num_chirps_per_frame)
        range_fft_windowed = range_fft * window_doppler[:, np.newaxis, np.newaxis]
        rd_map = fft(range_fft_windowed, axis=0)
        
        # [chirps, rx, samples] -> [samples, chirps, rx] 형태로 변환
        rd_map = np.transpose(rd_map, (2, 0, 1))
        
        return rd_map
    
    def generate_micro_doppler_spectrogram(self, sequence_path, start_frame, 
                                          num_frames, range_bin=None, 
                                          rx_antenna=0, window_size=32, 
                                          overlap=0.75):
        """
        마이크로 도플러 스펙트로그램 생성
        
        Args:
            sequence_path: 시퀀스 폴더 경로
            start_frame: 시작 프레임 인덱스
            num_frames: 분석할 프레임 개수
            range_bin: 관심 거리 빈 (None이면 자동 선택)
            rx_antenna: 사용할 Rx 안테나 인덱스
            window_size: STFT 윈도우 크기 (프레임 단위)
            overlap: 윈도우 오버랩 비율 (0~1)
            
        Returns:
            frequencies: 도플러 주파수 (Hz)
            times: 시간 축 (초)
            spectrogram: 스펙트로그램 [주파수, 시간]
            rd_maps: Range-Doppler 맵 리스트 (디버깅용)
        """
        print(f"Loading {num_frames} frames from {sequence_path}...")
        
        # 여러 프레임의 Range-Doppler 맵 생성
        rd_maps = []
        doppler_profiles = []
        
        for frame_offset in range(num_frames):
            frame_idx = start_frame + frame_offset
            
            try:
                # ADC 데이터 로드
                adc_data = self.load_adc_data(sequence_path, frame_idx)
                
                # Range-Doppler 맵 계산
                rd_map = self.compute_range_doppler(adc_data)
                rd_maps.append(rd_map)
                
                # 특정 range bin 선택
                if range_bin is None:
                    # 최대 에너지를 가진 range bin 자동 선택
                    power_profile = np.mean(np.abs(rd_map[:, :, rx_antenna])**2, axis=1)
                    range_bin = np.argmax(power_profile)
                
                # 선택된 range bin의 도플러 프로파일 추출
                doppler_profile = rd_map[range_bin, :, rx_antenna]
                doppler_profiles.append(doppler_profile)
                
                if frame_offset % 10 == 0:
                    print(f"  Processed {frame_offset}/{num_frames} frames...")
                    
            except Exception as e:
                print(f"Error loading frame {frame_idx}: {e}")
                continue
        
        # [num_frames, num_chirps] 형태로 변환
        doppler_data = np.array(doppler_profiles)
        print(f"Doppler data shape: {doppler_data.shape}")
        
        # 각 chirp에 대해 시간축 STFT 수행
        nperseg = int(window_size)
        noverlap = int(window_size * overlap)
        
        spectrograms_all_chirps = []
        
        for chirp_idx in range(self.num_chirps_per_frame):
            # 시간축 신호 추출
            time_series = doppler_data[:, chirp_idx]
            
            # STFT 계산 [web:45][web:48]
            f, t, Sxx = signal.spectrogram(
                time_series,
                fs=self.frame_rate,  # 프레임 레이트
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=256,  # Zero-padding
                return_onesided=False,
                mode='magnitude'
            )
            
            # Ensure Sxx is 2D (freq, time)
            if Sxx.ndim == 1:
                Sxx = Sxx[:, np.newaxis]
                
            spectrograms_all_chirps.append(Sxx)
        
        # 모든 chirp 평균
        spectrogram_avg = np.mean(spectrograms_all_chirps, axis=0)
        
        # Ensure spectrogram_avg is 2D
        if spectrogram_avg.ndim == 1:
            spectrogram_avg = spectrogram_avg[:, np.newaxis]
        
        # 주파수를 도플러 속도로 변환
        doppler_velocity = (f * self.wavelength) / 2
        
        # 주파수 중심 정렬
        sorted_idx = np.argsort(f)
        frequencies = f[sorted_idx]
        velocities = doppler_velocity[sorted_idx]
        spectrogram_sorted = spectrogram_avg[sorted_idx, :]
        
        print(f"Spectrogram shape: {spectrogram_sorted.shape}")
        print(f"Velocity range: {velocities.min():.2f} to {velocities.max():.2f} m/s")
        
        return frequencies, velocities, t, spectrogram_sorted, rd_maps
    
    def plot_micro_doppler(self, velocities, times, spectrogram, 
                          save_path=None, vmin=None, vmax=None):
        """
        마이크로 도플러 스펙트로그램 시각화
        
        Args:
            velocities: 속도 축 (m/s)
            times: 시간 축 (초)
            spectrogram: 스펙트로그램 데이터
            save_path: 저장 경로 (None이면 표시만)
            vmin, vmax: 컬러맵 범위
        """
        plt.figure(figsize=(12, 6))
        
        # dB 스케일 변환
        spectrogram_db = 20 * np.log10(spectrogram + 1e-10)
        
        if vmin is None:
            vmin = np.percentile(spectrogram_db, 5)
        if vmax is None:
            vmax = np.percentile(spectrogram_db, 95)
        
        plt.pcolormesh(times, velocities, spectrogram_db, 
                      shading='gouraud', cmap='jet', 
                      vmin=vmin, vmax=vmax)
        
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Doppler Velocity (m/s)', fontsize=12)
        plt.title('Micro-Doppler Spectrogram', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        else:
            plt.show()
    
    def plot_range_doppler_sequence(self, rd_maps, rx_antenna=0, save_dir=None):
        """
        여러 프레임의 Range-Doppler 맵 시각화
        
        Args:
            rd_maps: Range-Doppler 맵 리스트
            rx_antenna: 표시할 Rx 안테나 인덱스
            save_dir: 저장 디렉토리
        """
        num_frames = len(rd_maps)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 6개 프레임을 균등하게 샘플링
        frame_indices = np.linspace(0, num_frames-1, 6, dtype=int)
        
        for idx, (ax, frame_idx) in enumerate(zip(axes, frame_indices)):
            rd_map = rd_maps[frame_idx]
            rd_power = np.abs(rd_map[:, :, rx_antenna])**2
            rd_db = 10 * np.log10(rd_power + 1e-10)
            
            im = ax.imshow(rd_db.T, aspect='auto', cmap='jet', 
                          origin='lower', interpolation='bilinear')
            ax.set_title(f'Frame {frame_idx}')
            ax.set_xlabel('Range Bin')
            ax.set_ylabel('Doppler Bin')
            plt.colorbar(im, ax=ax, label='Power (dB)')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, 'range_doppler_sequence.png')
            plt.savefig(save_path, dpi=150)
            print(f"Saved RD sequence to {save_path}")
        else:
            plt.show()


# 사용 예제
if __name__ == "__main__":
    # 파라미터 설정
    calib_path = '/path/to/RADIal/SignalProcessing/CalibrationTable.npy'
    sequence_path = '/path/to/RADIal/sequences/RECORD@2019-09-16_12-31-03'
    
    # MicroDopplerGenerator 초기화
    md_gen = MicroDopplerGenerator(calib_path)
    
    # 마이크로 도플러 스펙트로그램 생성
    start_frame = 0
    num_frames = 100  # 100 프레임 분석 (5초 @ 20fps)
    
    freqs, vels, times, spectrogram, rd_maps = md_gen.generate_micro_doppler_spectrogram(
        sequence_path=sequence_path,
        start_frame=start_frame,
        num_frames=num_frames,
        range_bin=None,  # 자동 선택
        rx_antenna=0,
        window_size=32,
        overlap=0.75
    )
    
    # 시각화
    md_gen.plot_micro_doppler(vels, times, spectrogram, 
                              save_path='micro_doppler.png')
    
    # Range-Doppler 시퀀스 확인
    md_gen.plot_range_doppler_sequence(rd_maps[:30], rx_antenna=0,
                                       save_dir='./output')
