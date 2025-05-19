import sys
import os # 新增导入

import pybamm

import em2ecm.models.a123_88param as a123_88param
import em2ecm.models.a123_param as a123_param
from em2ecm.sensitivity_analysis import SensitivityAnalysis
from em2ecm.utils.logger import get_logger, set_logging_level

# --- 修改 sys.path 设置 ---
# 获取项目根目录的绝对路径
# __file__ 是当前脚本的路径
# os.path.dirname(__file__) 是当前脚本所在的目录 (em2ecm/sensitivity_analysis)
# os.path.join(os.path.dirname(__file__), "..", "..") 上溯两级到项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 如果项目根目录不在 sys.path 中，则添加它
# 这允许从项目根目录导入模块，例如 'scripts' 包或 'em2ecm' 包自身
if project_root not in sys.path:
    sys.path.insert(0, project_root) # 插入到开头以优先使用项目模块
# --- sys.path 设置结束 ---

from scripts import paths # 此导入现在应该可以工作

# 脚本其余部分保持不变
a123_param = pybamm.ParameterValues(a123_param.get_parameter_values()) # 使用 em2ecm.models.a123_param.get_parameter_values()

logger = get_logger(__name__) # 如果作为模块导入，__name__ 将是 'em2ecm.sensitivity_analysis.run_example'；如果直接运行，则是 '__main__'
if len(sys.argv) > 1 and sys.argv[1].lower() == "debug":
    set_logging_level("DEBUG")
    logger.info("Debug logging enabled.")

if __name__ == "__main__":
    # experiment_type_list = ["pulse-pulse", "cccv-cc", "cccv-pulse", "xftg"]
    experiment_type_list = ["cccv-cc"] # 示例中默认运行的实验类型

    for experiment_type in experiment_type_list:
        # category_list = ["geometry", "thermal", "electrochemical", "degradation"]
        category_list = ["geometry"] # 示例中默认运行的参数类别
        for category in category_list:
            sample_n = 2 # 默认采样数

            # 解析命令行参数
            if len(sys.argv) == 4: # python run_example.py category experiment_type sample_n
                category = sys.argv[1]
                experiment_type = sys.argv[2]
                sample_n = int(sys.argv[3])
            elif len(sys.argv) > 1 and sys.argv[1].lower() != "debug": # 参数数量不对且第一个参数不是 "debug"
                print(f"用法: python {os.path.basename(__file__)} [category experiment_type sample_n]")
                print(f"或: python {os.path.basename(__file__)} debug")
                sys.exit(1)

            tag = f"{category}_{experiment_type}_{sample_n}"
            
            # 确保 paths.results_dir 被正确解析
            # 对于示例，将结果保存在一个明确的子目录中
            save_dir = paths.results_dir / "parameter_sensitivity_example" / tag
            try:
                save_dir.mkdir(parents=True, exist_ok=True) # 确保保存目录存在
            except OSError as e:
                logger.error(f"创建目录 {save_dir} 失败: {e}")
                sys.exit(1)

            logger.info(f"开始敏感性分析: {tag}")
            logger.info(f"结果将保存到: {save_dir}")

            sa_param = a123_88param.get_parameter(
                category=category, updated=False, use_simple=True
            )
            sa_param_bounds = a123_88param.get_parameter_bounds(
                category=category, updated=False, use_simple=True
            )

            sa = SensitivityAnalysis(
                param=a123_param,
                sa_param=sa_param,
                sa_param_bounds=sa_param_bounds,
                save_dir=save_dir,
                experiment_type=experiment_type,
                category=category,
                sample_n=sample_n,
                use_current_results=False, # 对于新的示例运行，通常为 False
                use_mpi=False, # 对于简单示例，通常为 False
                aging_cycle_number=2, # 示例值
                aging_group_number=3, # 示例值
            )

            sa.run()
            logger.info(f"敏感性分析完成: {tag}")

    logger.info("所有示例敏感性分析已完成。")
