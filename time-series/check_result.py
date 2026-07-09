import os, argparse

SEED_NUM = 3

def main(result_data_path: str):
    
    result_dir = os.path.join(os.getcwd(), 'time-series', result_data_path)
    if not os.path.exists(result_dir):
        raise FileExistsError(f"Result dir:{result_dir} does not exist.")
    model_name = [f for f in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, f))]
    for model in model_name:
        model_dir_path = os.path.join(result_dir, model)
        task_name = [f for f in os.listdir(model_dir_path) if os.path.isdir(os.path.join(model_dir_path, f))]
        for task in task_name:
            task_dir_path = os.path.join(model_dir_path, task)
            dataset_name = [f for f in os.listdir(task_dir_path) if os.path.isdir(os.path.join(task_dir_path, f))]
            for dataset in dataset_name:
                dataset_dir_path = os.path.join(task_dir_path, dataset)
                mission_name = [f for f in os.listdir(dataset_dir_path) if os.path.isdir(os.path.join(dataset_dir_path, f))]
                for mission in mission_name:
                    mission_dir_path = os.path.join(dataset_dir_path, mission)
                    seed_name = [f for f in os.listdir(mission_dir_path) if os.path.isdir(os.path.join(mission_dir_path, f))]
                    if len(seed_name) != SEED_NUM:
                        print(f"{model} {task} {dataset} lack seed complete, all seed name: {seed_name}")
                    for seed in seed_name:
                        seed_path_dir = os.path.join(mission_dir_path, seed)
                        if task == 'finetune20': # finetune20
                            if "checkpoint_epoch_20.pth" not in os.listdir(seed_path_dir):
                                print(f"{model} {task} {dataset} {mission} {seed} not complete.")
                        elif task == "pretrain10": # pretrain10
                            if "jepa_pretrain_epoch_10.pth" not in os.listdir(seed_path_dir):
                                print(f"{model} {task} {dataset} {mission} {seed} not complete.")
                        elif task == "pretrain10_finetune10":
                            if "checkpoint_epoch_10.pth" not in os.listdir(seed_path_dir):
                                print(f"{model} {task} {dataset} {mission} {seed} not complete.")

if __name__ == '__main__':
    
    parsers = argparse.ArgumentParser()
    parsers.add_argument("--result_data_path", required=True, type=str, help="指定时间文件夹进行检查。")
    arg = parsers.parse_args()
    main(arg.result_data_path)