
# main.py
import argparse

from src.workflow import run_vision_workflow


def main():
    parser = argparse.ArgumentParser(
        description="壁画病害视觉分析"
    )

    parser.add_argument(
        "image",
        help="需要分析的图片路径"
    )

    args = parser.parse_args()

    try:
        result = run_vision_workflow(args.image)

        print("\n分析完成！")
        print("图片编号：", result["image_id"])
        print("分析摘要：", result["summary"])
        print("\n结构化结果：")

        for finding in result["findings"]:
            print(
                finding["disease"],
                finding["status"],
                finding["confidence"]
            )

        print("\nJSON 结果已保存到 output/results/")

    except Exception as e:
        print(f"\n分析失败：{e}")
        raise


if __name__ == "__main__":
    main()



# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
