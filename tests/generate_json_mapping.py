#!/usr/bin/env python3
"""
生成 JSON 格式的完整變數對應檔案
"""

import json
from pathlib import Path
from typing import Any


def translate_value_label(label: str) -> str:
    """翻譯回答選項"""
    translations = {
        # Common missing values
        "Valid Skip": "有效跳題",
        "Not Applicable": "不適用",
        "Invalid": "無效",
        "No Response": "未回答",
        # Yes/No
        "No, never": "否，從未",
        "Yes, once": "是，一次",
        "Yes, twice or more": "是，兩次或以上",
        # Frequency
        "Not at all": "完全沒有",
        "Very little": "很少",
        "To some extent": "某種程度",
        "A lot": "很多",
        # School location
        "A village, hamlet or rural area (fewer than 3 000 people)": "村莊、小村或鄉村地區（少於3000人）",
        "A small town (3 000 to about 15 000 people)": "小鎮（3000至約15000人）",
        "A town (15 000 to about 100 000 people)": "城鎮（15000至約100000人）",
        "A city (100 000 to about 1 000 000 people)": "城市（100000至約100萬人）",
        "A large city (1 000 000 to about 10 000 000 people)": "大城市（100萬至約1000萬人）",
        "A megacity (with over 10 000 000 people)": "超大城市（超過1000萬人）",
        # Work frequency
        "No work in household or care of family members": "不做家務或照顧家人",
    }

    # Check for pattern matches
    if "time" in label.lower() and "working in household" in label.lower():
        num = label.split()[0]
        return f"每週做家務或照顧家人{num}次"

    if "or more times of working" in label.lower():
        num = label.split()[0]
        return f"每週做家務或照顧家人{num}次或以上"

    return translations.get(label, label)


def translate_question(question: str) -> str:
    """翻譯問題文本"""
    translations = {
        "Home possessions (WLE)": "家庭財產指數（加權似然估計）",
        "Working in household/take care of family members before or after school": "上學前後在家做家務或照顧家人的頻率",
        "Instruction hindered by: A lack of teaching staff": "教學受阻因素：缺乏教學人員",
        "Instruction hindered by: Inadequate or poorly qualified teaching staff": "教學受阻因素：教學人員不足或素質不佳",
        "Instruction hindered by: A lack of assisting staff": "教學受阻因素：缺乏輔助人員",
        "Instruction hindered by: A lack of educational material (e.g. textbooks, IT equipment, library or laboratory material)": "教學受阻因素：缺乏教材（如教科書、IT設備、圖書館或實驗室材料）",
        "ICT Resources (WLE)": "資訊與通訊科技資源指數（加權似然估計）",
        "ICT availability outside of school  (WLE)": "校外資訊與通訊科技可用性（加權似然估計）",
        "Availability and Usage of ICT at Home": "家中資訊與通訊科技的可用性與使用情況",
        "Have you ever repeated a [grade]: At [ISCED 1]": "你是否曾經留級：小學階段（ISCED 1）",
        "Have you ever repeated a [grade]: At [ISCED 2]": "你是否曾經留級：國中階段（ISCED 2）",
        "Have you ever repeated a [grade]: At [ISCED 3]": "你是否曾經留級：高中階段（ISCED 3）",
        "Mathematics Anxiety (WLE)": "數學焦慮指數（加權似然估計）",
        "Which of the following definitions best describes the community in which your school is located?": "以下哪個定義最能描述你學校所在的社區？",
        "Have you ever missed school for more than three months in a row: At [ISCED 1]": "你是否曾經連續缺課超過三個月：小學階段（ISCED 1）",
        "Have you ever missed school for more than three months in a row: At [ISCED 2]": "你是否曾經連續缺課超過三個月：國中階段（ISCED 2）",
        "Have you ever missed school for more than three months in a row: At [ISCED 3]": "你是否曾經連續缺課超過三個月：高中階段（ISCED 3）",
    }

    return translations.get(question, question)


def generate_json_mapping() -> dict[str, Any]:
    """生成完整的 JSON 變數對應"""

    # 載入 metadata
    stu_meta = json.load(Path("/data/CY08MSP_STU_QQQ_metadata.json").open())
    sch_meta = json.load(Path("/data/CY08MSP_SCH_QQQ_metadata.json").open())

    # 定義變數分組
    variable_list = {
        "access_to_resources": [
            ("HOMEPOS", "Student", stu_meta),
            ("WORKHOME", "Student", stu_meta),
            ("SC017Q01NA", "School", sch_meta),
            ("SC017Q02NA", "School", sch_meta),
            ("SC017Q03NA", "School", sch_meta),
            ("SC017Q05NA", "School", sch_meta),
        ],
        "internet_access": [
            ("ICTRES", "Student", stu_meta),
            ("ICTHOME", "Student", stu_meta),
            ("ICTAVHOM", "Student", stu_meta),
        ],
        "learning_difficulties": [
            ("ST127Q01TA", "Student", stu_meta),
            ("ST127Q02TA", "Student", stu_meta),
            ("ST127Q03TA", "Student", stu_meta),
            ("ANXMAT", "Student", stu_meta),
        ],
        "geographic_disadvantage": [
            ("SC001Q01TA", "School", sch_meta),
            ("ST260Q01JA", "Student", stu_meta),
            ("ST260Q02JA", "Student", stu_meta),
            ("ST260Q03JA", "Student", stu_meta),
        ],
    }

    result = {
        "metadata": {
            "generated_date": "2025-12-09",
            "pisa_cycle": "2022",
            "total_dimensions": 4,
            "total_variables": 18,
        },
        "dimensions": {},
    }

    for dimension, var_list in variable_list.items():
        dimension_data: dict[str, Any] = {"variables": {}}

        for var_name, source, metadata in var_list:
            if var_name in metadata:
                var_data = metadata[var_name]

                # 翻譯 value labels
                translated_values = {}
                for key, val in var_data.get("value_labels", {}).items():
                    translated_values[key] = {
                        "original": val,
                        "chinese": translate_value_label(val),
                    }

                dimension_data["variables"][var_name] = {
                    "source": source,
                    "data_type": var_data.get("type", "unknown"),
                    "question": {
                        "original": var_data.get("label", ""),
                        "chinese": translate_question(var_data.get("label", "")),
                    },
                    "value_labels": translated_values,
                    "sample_values": var_data.get("sample_values", []),
                }

        result["dimensions"][dimension] = dimension_data

    return result


def main() -> None:
    """主程式"""
    print("生成 JSON 變數對應檔案...\n")

    json_data = generate_json_mapping()

    # 儲存檔案
    output_path = Path("/home/jovyan/workspace/artifacts/pisa_variable_mapping.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"✅ JSON 檔案已生成: {output_path}")
    print(f"📏 檔案大小: {output_path.stat().st_size / 1024:.2f} KB")

    # 統計資訊
    print("\n📊 統計資訊:")
    print(f"  維度數量: {json_data['metadata']['total_dimensions']}")
    print(f"  變數總數: {json_data['metadata']['total_variables']}")

    print("\n各維度變數數量:")
    for dim, data in json_data["dimensions"].items():
        print(f"  {dim}: {len(data['variables'])} 個變數")

    # 顯示範例
    print("\n" + "=" * 80)
    print("範例變數 (ST127Q01TA):")
    print("=" * 80)
    example = json_data["dimensions"]["learning_difficulties"]["variables"]["ST127Q01TA"]
    print(json.dumps(example, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
