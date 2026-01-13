# kineticEQ/analysis/BGK1D/benchmark.py
from kineticEQ import Config, Engine, BGK1D
import torch

class BGK1DBenchmark:
    def __init__(self, config: Config):
        self.config = config
        self.engine = Engine(config)
    
    # ベンチマークメソッド
    def run_benchmark(self,
                      benc_type="spatial",
                      grid_list=[16, 32, 64, 128, 256, 512, 1024],
                      nv_list=[64, 128, 256]):
        print(f"--- Benchmark Start, benc_type: {benc_type} ---")

        # 結果保存用辞書
        self.benchmark_results = {
            'bench_type' : benc_type,
        }

        if benc_type == "spatial":
            self._run_benchmark_space(grid_list)
        elif benc_type == "velocity":
            self._run_benchmark_velocity(grid_list)
        elif benc_type == "time":
            self._run_benchmark_time(grid_list, nv_list)
        else:
            raise ValueError(f"Unknown benchmark type: {benc_type}")

        #self.plot_benchmark_comparison(benc_type)

        print("--- Benchmark Completed ---")
        return self.benchmark_results

    #空間差分ベンチマーク
    def _run_benchmark_space(self, grid_list):

        # grid_listに従って空間差分ベンチマークを実行
        for bench_iter in range(len(grid_list)):
            self._runbench_spatial(grid_list[bench_iter])
            n, u, T = self.calculate_moments()
            self.benchmark_results[grid_list[bench_iter]] = {
                'x': self.x.cpu().numpy().copy(),   
                'v': self.v.cpu().numpy().copy(),    
                'f': self.f.cpu().numpy().copy(), 
                'n': n.cpu().numpy().copy(),
                'u': u.cpu().numpy().copy(),
                'T': T.cpu().numpy().copy(),
                'dx': self.dx,
                'dv': self.dv
            }

    #速度差分ベンチマーク
    def _run_benchmark_velocity(self, grid_list):

        # grid_listに従って速度差分ベンチマークを実行
        for bench_iter in range(len(grid_list)):
            self._runbench_velocity(grid_list[bench_iter])
            n, u, T = self.calculate_moments()
            self.benchmark_results[grid_list[bench_iter]] = {
                'x': self.x.cpu().numpy().copy(),
                'v': self.v.cpu().numpy().copy(),    
                'f': self.f.cpu().numpy().copy(), 
                'n': n.cpu().numpy().copy(),
                'u': u.cpu().numpy().copy(),
                'T': T.cpu().numpy().copy(),
                'dv': self.dv,
                'dx': self.dx  
            }

    # 実行時間ベンチマーク
    def _run_benchmark_time(self, nx_list, nv_list):

        # 実行中の計算デバイス名の取得
        device_name = torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else str(self.device)
        self.benchmark_results['device_name'] = device_name
        
        # CPU名を取得
        import platform
        cpu_name = None
        
        # 方法1: /proc/cpuinfo から取得（Linux）
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_name = line.split(':')[1].strip()
                        break
        except:
            pass
        
        # 方法2: cpuinfo ライブラリを使用
        if not cpu_name:
            try:
                import cpuinfo
                cpu_name = cpuinfo.get_cpu_info()['brand_raw']
            except:
                pass
        
        # 方法3: platform.processor()
        if not cpu_name:
            cpu_name = platform.processor()
        
        # フォールバック: アーキテクチャ名
        if not cpu_name:
            cpu_name = f"{platform.machine()} CPU"
        
        self.benchmark_results['cpu_name'] = cpu_name
        
        # 結果保存用辞書を初期化
        self.benchmark_results['timing_results'] = {}

        # grid_listに従って実行時間ベンチマークのグリッドテストを実行
        for nx in nx_list:
            for nv in nv_list:
                grid_key = f"{nx}x{nv}"
                print(f"--- Test Grid: {grid_key} ---")

                # グリッドテストベンチマークの実行と結果保存
                timing_result = self._runbench_time(nx, nv)
                self.benchmark_results['timing_results'][grid_key] = timing_result

        return self.benchmark_results

    # 空間分割数ベンチマーク用シミュレーションメソッド
    def _runbench_spatial(self, nx_num):
        print(f"--- Run Benchmark Simulation (nx = {nx_num}) ---")

        # パラメータ設定
        self.nx = nx_num
        self.dx = self.Lx / (self.nx - 1)
        self.nt = int(self.T_total / self.dt)

        # 配列確保
        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        # プログレスバーを初期化
        with get_progress_bar(self.use_tqdm,total=self.nt, desc="Progress", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for step in range(self.nt):
                if self.solver == "explicit":
                    if self.explicit_solver == 'backend':
                        self._explicit_update_cuda_backend()
                    else:
                        self._explicit_update()
                elif self.solver == "implicit":
                    if self.implicit_solver == 'backend':
                        self._implicit_update_cuda_backend()
                    elif self.implicit_solver == 'holo':
                        self._implicit_update_holo()
                    else:
                        self._implicit_cusolver_update()
                    
                # 配列交換
                self.f, self.f_new = self.f_new, self.f
                pbar.update(1)

    # 速度分割数ベンチマーク用シミュレーションメソッド
    def _runbench_velocity(self, nv_num):
        print(f"--- Run Benchmark Simulation (nv = {nv_num}) ---")

        # パラメータ設定
        self.nv = nv_num
        self.dv = 2 * self.v_max / (self.nv - 1)
        self.nt = int(self.T_total / self.dt)

        # 配列確保
        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        # プログレスバーを初期化
        with get_progress_bar(self.use_tqdm,total=self.nt, desc="Progress", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:

            for step in range(self.nt):
                if self.solver == "explicit":
                    if self.explicit_solver == 'backend':
                        self._explicit_update_cuda_backend()
                    else:
                        self._explicit_update()
                elif self.solver == "implicit":
                    if self.implicit_solver == 'backend':
                        self._implicit_update_cuda_backend()
                    elif self.implicit_solver == 'holo':
                        self._implicit_update_holo()
                    else:
                        self._implicit_cusolver_update()
                    
                # 配列交換
                self.f, self.f_new = self.f_new, self.f
                pbar.update(1)

    # 実行時間ベンチマーク用シミュレーションメソッド
    def _runbench_time(self, nx_num, nv_num):
        import time
        
        self.nx = nx_num
        self.nv = nv_num
        self.dx = self.Lx / (self.nx - 1)
        self.dv = 2 * self.v_max / (self.nv - 1)

        self.Array_allocation()
        self.set_initial_condition()
        self.apply_boundary_condition()

        # CUDA時間測定の準備
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        # Warm-up: 初回JITコンパイルとメモリ確保のオーバーヘッドを除去
        warmup_steps = min(5, self.nt // 4)  # 最大5ステップ、全体の1/4まで
        for step in range(warmup_steps):
            if self.solver == "explicit":
                # self._explicit_update()
                if self.explicit_solver == 'backend':
                    self._explicit_update_cuda_backend()
                else:
                    self._explicit_update()
            elif self.solver == "implicit":
                if self.implicit_solver == 'backend':
                    self._implicit_update_cuda_backend()
                elif self.implicit_solver == 'holo':
                    self._implicit_update_holo()
                else:
                    self._implicit_cusolver_update()
                
            # 配列交換
            self.f, self.f_new = self.f_new, self.f
        
        # 本計測開始
        cpu_start_time = time.perf_counter()
        
        # GPU時間測定開始
        if self.device.type == 'cuda':
            start_event.record()

        # ベンチマーク用ループ（プログレスバー無し）
        for step in range(self.nt):
            if self.solver == "explicit":
                # self._explicit_update()
                if self.explicit_solver == 'backend':
                    self._explicit_update_cuda_backend()
                else:
                    self._explicit_update()
            elif self.solver == "implicit":
                if self.implicit_solver == 'backend':
                    self._implicit_update_cuda_backend()
                elif self.implicit_solver == 'holo':
                    self._implicit_update_holo()
                else:
                    self._implicit_cusolver_update()

            # 配列交換
            self.f, self.f_new = self.f_new, self.f

        # GPU完了を待ってから時間測定終了
        if self.device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()  # GPU完了を保証
        
        cpu_total_time = time.perf_counter() - cpu_start_time
        
        # 結果をまとめる
        timing_result = {
            'nx': nx_num,
            'nv': nv_num,
            'total_grid_points': nx_num * nv_num,
            'device': str(self.device),
            'total_steps': self.nt,
            'warmup_steps': warmup_steps,
            'cpu_total_time_sec': cpu_total_time,  # GPU計算を包含した壁時計時間
        }
        
        # CUDA時間も記録（純粋なGPU計算時間）
        if self.device.type == 'cuda':
            gpu_total_time_ms = start_event.elapsed_time(end_event)
            timing_result.update({
                'gpu_total_time_ms': gpu_total_time_ms,
                'gpu_total_time_sec': gpu_total_time_ms / 1000,
            })
        
        return timing_result