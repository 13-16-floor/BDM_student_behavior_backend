#!/usr/bin/env python3
"""
PISA Variable Validation Script

驗證對話中提到的所有變數是否存在於 metadata 中，
並生成包含問題文本和回答選項的完整對應表。
"""

import json
from pathlib import Path
from typing import Any


class PISAVariableValidator:
    """PISA 變數驗證器"""

    def __init__(self, metadata_dir: str = "/data"):
        self.metadata_dir = Path(metadata_dir)
        self.metadata: dict[str, Any] = {}
        self.load_all_metadata()

    def load_all_metadata(self) -> None:
        """載入所有 metadata JSON 檔案"""
        metadata_files = list(self.metadata_dir.glob("*_metadata.json"))
        print(f"找到 {len(metadata_files)} 個 metadata 檔案:\n")

        for filepath in metadata_files:
            source_name = filepath.stem.replace("_metadata", "")
            print(f"  - {filepath.name}")

            with filepath.open(encoding="utf-8") as f:
                data = json.load(f)
                self.metadata[source_name] = data

        print(f"\n總共載入 {len(self.metadata)} 個資料來源\n")

    def find_variable(self, var_name: str) -> dict[str, Any]:
        """
        在所有 metadata 中搜尋變數

        Returns:
            {
                'found': bool,
                'source': str,
                'data': dict  # 完整的變數資訊
            }
        """
        for source_name, metadata in self.metadata.items():
            # 檢查變數是否在 metadata 的 keys 中
            if var_name in metadata:
                return {
                    "found": True,
                    "source": source_name,
                    "data": metadata[var_name],
                }

        return {"found": False, "source": None, "data": None}

    def validate_dimension_variables(self) -> dict[str, Any]:
        """驗證四個維度的所有變數"""

        # 定義四個維度的變數列表（根據對話內容）
        dimensions: dict[str, dict[str, Any]] = {
            "Access_to_Resources": {
                "description": "家庭和學校的教育資源可及性",
                "variables": {
                    "home_level": [
                        "ST011Q01TA",  # Desk to study at
                        "ST011Q02TA",  # Quiet place to study
                        "ST011Q03TA",  # Computer for school work
                        "ST011Q04TA",  # Educational software
                        "ST011Q05TA",  # Books to help with school work
                        "ST011Q06TA",  # Technical reference books
                        "ST011Q16NA",  # Classic literature
                        "ST013Q01TA",  # Number of books at home
                        "HEDRES",  # Home educational resources (composite)
                        "HOMEPOS",  # Home possessions (SES indicator)
                        "WORKHOME",  # Added - workspace at home
                    ],
                    "school_level": [
                        "SC017Q01NA",  # Shortage of science lab equipment
                        "SC017Q02NA",  # Shortage of instructional materials
                        "SC017Q03NA",  # Shortage of computers
                        "SC017Q05NA",  # Shortage of library materials
                    ],
                },
            },
            "Internet_Access": {
                "description": "數位資源與網路連線可及性",
                "variables": {
                    "home_ict": [
                        "ST011Q03TA",  # Computer (重複但重要)
                        "ST011Q04TA",  # Educational software
                        "ST012Q01TA",  # Number of computers
                        "ST012Q02TA",  # Number of tablets
                        "ST012Q06NA",  # E-book reader
                        "ST012Q09NA",  # Internet connection
                        "ICTRES",  # ICT resources composite
                        "ICTHOME",  # Added - ICT at home
                        "ICTAVHOM",  # Added - ICT availability at home
                    ],
                    "ict_usage": [
                        "IC001Q01TA",  # ICT availability outside school
                        "IC009Q01TA",  # Use ICT for school tasks
                        "IC010Q01TA",  # Use ICT for leisure
                    ],
                },
            },
            "Learning_Disabilities": {
                "description": "學習困難與障礙（間接測量）",
                "variables": {
                    "grade_repetition": [
                        "ST127Q01TA",  # Repeated grade in primary
                        "ST127Q02TA",  # Repeated grade in lower secondary
                        "ST127Q03TA",  # Repeated grade in upper secondary
                    ],
                    "learning_anxiety": [
                        "ST118Q01NA",  # Math anxiety
                        "ST118Q02NA",  # Math anxiety items
                        "ST118Q03NA",
                        "ST118Q04NA",
                        "ST118Q05NA",
                        "ANXMAT",  # Math anxiety composite
                    ],
                    "self_efficacy": [
                        "ST119Q01NA",  # Math self-efficacy
                        "ST119Q02NA",
                        "ST119Q03NA",
                        "ST119Q04NA",
                        "ANXMAT",  # Added - anxiety composite
                    ],
                },
            },
            "Geographic_Disadvantage": {
                "description": "地理位置與通勤距離（間接測量）",
                "variables": {
                    "school_location": [
                        "SC001Q01TA",  # School location (rural/urban)
                    ],
                    "student_level_proxies": [
                        "ST260Q01JA",  # Student-level geographic proxy 1
                        "ST260Q02JA",  # Student-level geographic proxy 2
                        "ST260Q03JA",  # Student-level geographic proxy 3
                    ],
                },
            },
        }

        # 驗證結果
        results: dict[str, Any] = {
            "summary": {"total_variables": 0, "found": 0, "missing": 0},
            "dimensions": {},
        }

        print("=" * 80)
        print("PISA 變數驗證報告")
        print("=" * 80 + "\n")

        for dimension, info in dimensions.items():
            print(f"\n## {dimension}")
            print(f"說明: {info['description']}\n")

            dimension_result = {
                "description": info["description"],
                "categories": {},
                "statistics": {"total": 0, "found": 0, "missing": 0},
            }

            for category, var_list in info["variables"].items():
                print(f"### {category}")
                category_result: dict[str, Any] = {"variables": {}}

                for var in var_list:
                    search_result = self.find_variable(var)
                    results["summary"]["total_variables"] += 1
                    dimension_result["statistics"]["total"] += 1

                    if search_result["found"]:
                        status = "✅"
                        results["summary"]["found"] += 1
                        dimension_result["statistics"]["found"] += 1

                        # 提取變數詳細資訊
                        var_data = search_result["data"]
                        category_result["variables"][var] = {
                            "found": True,
                            "source": search_result["source"],
                            "column_label": var_data.get("label", ""),
                            "data_type": var_data.get("type", ""),
                            "question_text": var_data.get("question", ""),
                            "value_labels": var_data.get("value_labels", {}),
                            "missing_values": var_data.get("missing", {}),
                        }

                        print(f"  {status} {var}")
                        print(f"      來源: {search_result['source']}")
                        print(f"      標籤: {var_data.get('label', 'N/A')}")

                        # 顯示問題文本（如果有）
                        if var_data.get("question"):
                            q_text = var_data["question"][:100]
                            print(f"      問題: {q_text}...")

                    else:
                        status = "❌"
                        results["summary"]["missing"] += 1
                        dimension_result["statistics"]["missing"] += 1

                        category_result["variables"][var] = {
                            "found": False,
                            "reason": "變數在 metadata 中不存在",
                        }

                        print(f"  {status} {var} - 未找到")

                print()
                dimension_result["categories"][category] = category_result

            results["dimensions"][dimension] = dimension_result

        # 輸出統計摘要
        print("\n" + "=" * 80)
        print("統計摘要")
        print("=" * 80)
        print(f"總變數數量: {results['summary']['total_variables']}")
        print(
            f"找到: {results['summary']['found']} ({results['summary']['found'] / results['summary']['total_variables'] * 100:.1f}%)"
        )
        print(
            f"缺失: {results['summary']['missing']} ({results['summary']['missing'] / results['summary']['total_variables'] * 100:.1f}%)"
        )
        print()

        # 按維度統計
        print("\n各維度統計:")
        for dim_name, dim_data in results["dimensions"].items():
            stats = dim_data["statistics"]
            print(
                f"  {dim_name}: {stats['found']}/{stats['total']} "
                f"({stats['found'] / stats['total'] * 100:.1f}%)"
            )

        return results

    def export_to_json(self, results: dict[str, Any], output_path: str) -> None:
        """匯出結果為 JSON 檔案"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n結果已匯出至: {output_file}")
        print(f"檔案大小: {output_file.stat().st_size / 1024:.2f} KB")


def main() -> None:
    """主程式"""
    print("\n" + "=" * 80)
    print("PISA 2022 變數驗證工具")
    print("=" * 80 + "\n")

    # 初始化驗證器
    validator = PISAVariableValidator(metadata_dir="/data")

    # 執行驗證
    results = validator.validate_dimension_variables()

    # 匯出結果
    output_path = "/home/jovyan/workspace/artifacts/pisa_variable_validation.json"
    validator.export_to_json(results, output_path)

    # 建議
    print("\n" + "=" * 80)
    print("建議")
    print("=" * 80)

    missing_count = results["summary"]["missing"]
    if missing_count > 0:
        print(f"\n⚠️  發現 {missing_count} 個變數不存在")
        print("\n建議處理方式:")
        print("1. 檢查變數名稱拼寫（可能是年份差異）")
        print("2. 使用替代變數（例如用 HOMEPOS 替代 HEDRES）")
        print("3. 從分析中移除缺失變數")
        print("4. 查閱 PISA 2022 codebook 確認正確變數名")
    else:
        print("\n✅ 所有變數都已找到！可以直接使用。")

    print("\n" + "=" * 80)
    print("驗證完成")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
