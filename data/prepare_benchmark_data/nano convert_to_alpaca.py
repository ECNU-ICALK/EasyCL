import json
import os
import argparse

def convert_squad_to_alpaca(dataset_dir, output_dir, json_files):
    """ 仅转换指定的 5 个数据集（train + test）为 Alpaca 格式 """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for json_file in json_files:
        json_path = os.path.join(dataset_dir, json_file)
        output_path = os.path.join(output_dir, json_file.replace(".json", "_alpaca.json"))

        print(f"\n🔄 正在转换: {json_file} -> {output_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                squad_data = json.load(f)

            alpaca_data = []

            for entry in squad_data["data"]:
                for paragraph in entry["paragraphs"]:
                    context = paragraph["context"]
                    for qa in paragraph["qas"]:
                        question = qa["question"]
                        label = qa["answers"][0]["text"] if qa["answers"] else "Unknown"

                        alpaca_entry = {
                            "instruction": question,  # 直接用 question 作为指令
                            "input": context,  # 用 context 作为输入
                            "output": label  # 类别标签作为输出
                        }
                        alpaca_data.append(alpaca_entry)

            # 保存转换后的数据
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(alpaca_data, f, indent=4, ensure_ascii=False)

            print(f"✅ 转换完成: {output_path}")

        except Exception as e:
            print(f"❌ 处理 {json_file} 时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="仅转换指定的 5 个数据集（train + test）为 Alpaca 格式")
    parser.add_argument("--path", type=str, default="/tmp/ycai015/datasets/LLMCL/data/", help="数据集目录")
    parser.add_argument("--output", type=str, default="/tmp/ycai015/datasets/LLMCL/alpaca_data/", help="输出目录")
    parser.add_argument("--files", type=str, nargs="+", default=[
        "ag_to_squad-train-v2.0.json",
        "ag_to_squad-test-v2.0.json",
        "amazon_to_squad-train-v2.0.json",
        "amazon_to_squad-test-v2.0.json",
        "dbpedia_to_squad-train-v2.0.json",
        "dbpedia_to_squad-test-v2.0.json",
        "yahoo_to_squad-train-v2.0.json",
        "yahoo_to_squad-test-v2.0.json",
        "yelp_to_squad-train-v2.0.json",
        "yelp_to_squad-test-v2.0.json"
    ], help="要转换的 JSON 文件")

    args = parser.parse_args()
    convert_squad_to_alpaca(args.path, args.output, args.files)

    print("\n🎉 指定的 5 个数据集转换完成！")
