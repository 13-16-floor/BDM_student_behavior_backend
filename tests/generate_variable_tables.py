#!/usr/bin/env python3
"""
生成完整的變數對應表格，包含原始問題和中文翻譯
"""

import json
from pathlib import Path


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
        # e.g., "1 time of working in household..."
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


def format_value_labels(value_labels: dict[str, str], is_continuous: bool = False) -> str:
    """格式化回答選項"""
    if is_continuous:
        # 對於連續變數，只顯示有效範圍
        valid_range = []
        for key, _val in value_labels.items():
            if float(key) < 90:  # 排除缺失值代碼
                valid_range.append(float(key))

        if valid_range:
            return f"連續變數（範圍：{min(valid_range):.2f} - {max(valid_range):.2f}）"
        return "連續變數"

    # 對於類別變數，顯示所有選項
    options = []
    for key, val in sorted(value_labels.items(), key=lambda x: float(x[0])):
        k_float = float(key)
        if k_float < 90:  # 只顯示有效值，排除缺失值代碼
            translated = translate_value_label(val)
            options.append(f"{int(k_float)}={translated}")

    return ", ".join(options) if options else "類別變數"


def generate_markdown_table() -> str:
    """生成完整的 Markdown 表格"""

    # 載入 metadata
    stu_meta = json.load(Path("/data/CY08MSP_STU_QQQ_metadata.json").open())
    sch_meta = json.load(Path("/data/CY08MSP_SCH_QQQ_metadata.json").open())

    # 定義變數分組
    dimensions = {
        "Access to Resources (資源可及性)": {
            "variables": [
                ("HOMEPOS", "Student", stu_meta, True),
                ("WORKHOME", "Student", stu_meta, False),
                ("SC017Q01NA", "School", sch_meta, False),
                ("SC017Q02NA", "School", sch_meta, False),
                ("SC017Q03NA", "School", sch_meta, False),
                ("SC017Q05NA", "School", sch_meta, False),
            ]
        },
        "Internet Access (網路與數位資源)": {
            "variables": [
                ("ICTRES", "Student", stu_meta, True),
                ("ICTHOME", "Student", stu_meta, True),
                ("ICTAVHOM", "Student", stu_meta, False),
            ]
        },
        "Learning Difficulties (學習困難)": {
            "variables": [
                ("ST127Q01TA", "Student", stu_meta, False),
                ("ST127Q02TA", "Student", stu_meta, False),
                ("ST127Q03TA", "Student", stu_meta, False),
                ("ANXMAT", "Student", stu_meta, True),
            ]
        },
        "Geographic Disadvantage (地理劣勢)": {
            "variables": [
                ("SC001Q01TA", "School", sch_meta, False),
                ("ST260Q01JA", "Student", stu_meta, False),
                ("ST260Q02JA", "Student", stu_meta, False),
                ("ST260Q03JA", "Student", stu_meta, False),
            ]
        },
    }

    output = []
    output.append("# PISA 2022 變數完整對應表\n")
    output.append("**包含原始問題與中文翻譯**\n\n")
    output.append("---\n\n")

    for idx, (dim_name, dim_data) in enumerate(dimensions.items(), 1):
        output.append(f"## {idx}️⃣ {dim_name}\n\n")

        # 表格標題
        output.append("| 變數名稱 | 來源 | 原始問題 | 中文翻譯 | 回答選項 |")
        output.append("|---------|------|----------|----------|----------|")

        for var_name, source, metadata, is_continuous in dim_data["variables"]:
            if var_name in metadata:
                var_data = metadata[var_name]

                # 提取資訊
                original_label = var_data.get("label", "N/A")
                chinese_label = translate_question(original_label)
                value_labels = var_data.get("value_labels", {})
                formatted_options = format_value_labels(value_labels, is_continuous)

                # 格式化輸出（處理多行）
                original_label_short = (
                    original_label[:60] + "..." if len(original_label) > 60 else original_label
                )
                chinese_label_short = (
                    chinese_label[:60] + "..." if len(chinese_label) > 60 else chinese_label
                )
                formatted_options_short = (
                    formatted_options[:80] + "..."
                    if len(formatted_options) > 80
                    else formatted_options
                )

                output.append(
                    f"| **{var_name}** | {source} | {original_label_short} | {chinese_label_short} | {formatted_options_short} |"
                )

        output.append("\n")

    # 詳細說明
    output.append("---\n\n")
    output.append("## 📝 詳細變數說明\n\n")

    for idx, (dim_name, dim_data) in enumerate(dimensions.items(), 1):
        output.append(f"### {idx}. {dim_name}\n\n")

        for var_name, source, metadata, _is_continuous in dim_data["variables"]:
            if var_name in metadata:
                var_data = metadata[var_name]

                output.append(f"#### `{var_name}`\n\n")
                output.append(f"**來源**: {source} Questionnaire\n\n")
                output.append(f"**原始問題**: {var_data.get('label', 'N/A')}\n\n")
                output.append(
                    f"**中文翻譯**: {translate_question(var_data.get('label', 'N/A'))}\n\n"
                )
                output.append(f"**資料類型**: {var_data.get('type', 'N/A')}\n\n")

                value_labels = var_data.get("value_labels", {})
                if value_labels:
                    output.append("**回答選項**:\n\n")
                    output.append("| 代碼 | 原始標籤 | 中文翻譯 |\n")
                    output.append("|------|----------|----------|\n")

                    for key, val in sorted(value_labels.items(), key=lambda x: float(x[0])):
                        translated = translate_value_label(val)
                        output.append(f"| {key} | {val} | {translated} |\n")
                    output.append("\n")

                output.append("---\n\n")

    return "\n".join(output)


def main() -> None:
    """主程式"""
    print("生成完整變數對應表...\n")

    markdown_content = generate_markdown_table()

    # 儲存檔案
    output_path = Path("/home/jovyan/workspace/artifacts/PISA_Variable_Complete_Tables.md")
    output_path.write_text(markdown_content, encoding="utf-8")

    print(f"✅ 完整表格已生成: {output_path}")
    print(f"📏 檔案大小: {output_path.stat().st_size / 1024:.2f} KB")

    # 同時輸出到螢幕（前100行）
    print("\n" + "=" * 80)
    print("預覽（前100行）:")
    print("=" * 80 + "\n")
    lines = markdown_content.split("\n")
    for line in lines[:100]:
        print(line)

    if len(lines) > 100:
        print(f"\n... ({len(lines) - 100} 行省略) ...")


if __name__ == "__main__":
    main()
