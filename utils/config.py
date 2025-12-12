import os 

# --------------------------------------------------------------------------------------
# 模仿附件中的 get_config() 函数，封装全局配置参数
# --------------------------------------------------------------------------------------
def get_config():
    """Return configuration parameters for the LinesDetection process."""
    
    # ####################################### 参数设置 #######################################
    # 使用一个字典来存储所有参数
    config = {
        'workspace': r"/home/rylynn/Pictures/LinesDetection_Workspace/datasets/Dublin/block2/",
        'n_jobs': 3,
        'num_H': 100,
        'random_contrast': True,
        # 分块参数
        'patch_size': 2048,
        'overlap': 512,
        # 线段保留
        'retain_ratio': 0.3
    }
    
    print("Configuration parameters:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("==============================================================================")
    
    return config

# --------------------------------------------------------------------------------------
# 模仿附件中的 PathManager 类，管理所有路径
# --------------------------------------------------------------------------------------
class PathManager:
    """Manages all file paths for the LinesDetection process."""
    
    def __init__(self, workspace_path):
        self.workspace = workspace_path
        
        # ####################################### 路径设置 #######################################
        # 使用 @property 属性来定义路径
        
    @property
    def sparse_model_path(self):
        return os.path.join(self.workspace, 'sparse')
    
    @property
    def images_path(self):
        return os.path.join(self.workspace, 'images')
    
    @property
    def depth_path(self):
        return os.path.join(self.workspace, 'depth_maps')
    
    @property
    def output_path(self):
        # 这一步会自动创建文件夹，但为了保持类属性的纯粹性，我们可以在初始化或外部调用时创建
        path = os.path.join(self.workspace, 'intermediate_results')
        # os.makedirs(path, exist_ok=True) # 外部调用时创建，保持类属性纯净
        return path
    
    @property
    def gt_path(self):
        path = os.path.join(self.workspace, 'gt_patches')
        # os.makedirs(path, exist_ok=True)
        return path
    
    
    @property
    def avg_vis_path(self):
        # 存放中间平均结果的可视化
        path = os.path.join(self.output_path, "avg_blocks")
        # os.makedirs(path, exist_ok=True)
        return path
    
    @property
    def final_vis_path(self):
        # 存放最终GT的可视化
        path = os.path.join(self.output_path, "visualize_blocks")
        # os.makedirs(path, exist_ok=True)
        return path
        
    def create_paths(self):
        """Creates all necessary directories based on the defined paths."""
        # 集合所有需要创建的目录路径
        paths_to_create = [
            self.output_path,
            self.gt_path,
            self.avg_vis_path,
            self.final_vis_path
        ]
        
        for path in paths_to_create:
            os.makedirs(path, exist_ok=True)

# --- 示例用法 (可选，用于测试) ---
if __name__ == '__main__':
    
    # 1. 获取配置
    config = get_config()
    
    # 2. 初始化路径管理器
    workspace = config['workspace']
    pm = PathManager(workspace)
    
    # 3. 创建目录
    print("\nAttempting to create directories:")
    pm.create_paths()
    
    # 4. 打印一些路径
    print("\nExample Paths:")
    print(f"Sparse Model: {pm.sparse_model_path}")
    print(f"GT HDF5: {pm.gt_hdf5_path}")
    print(f"Final Visualization: {pm.final_vis_path}")