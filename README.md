# BDM_student_behavior_backend

## Setup Instructions

### Docker Setup

啟動 Docker 容器服務：

```bash
# 使用現代 Docker Compose 語法（推薦）
docker compose up -d --build

# 或使用傳統 docker-compose 語法
docker-compose up -d --build
```

> **注意**: `docker compose`（無連字號）是現代 Docker CLI 的一部分，而 `docker-compose`（有連字號）是舊版獨立工具。兩者功能相同，使用其中任一即可。

服務啟動後，查看 JupyterLab logs 找到登入連結：

```bash
# 使用現代語法
docker compose logs jupyter

# 或使用傳統語法
docker-compose logs jupyter
```

### VS Code Remote Container Development

為方便開發，建議使用 VS Code Remote Container 功能：

1. 點擊 VS Code 左下角的綠色圖示
2. 選擇「Attach to Running Container」
3. 選擇 `bdm_student_behavior_backend_jupyter_1` 容器
4. 在遠端連接中，開啟 `/workspace` 目錄

進入容器後，執行以下命令執行應用：

```bash
# 打開 conda 環境
conda activate python310

# 執行應用查看分析結果
python -m behavior_analysis
```

## Development Setup

### 安裝開發依賴

本專案使用 [uv](https://github.com/astral-sh/uv) 作為 Python 套件管理工具。

```bash
# 安裝所有依賴（包含開發依賴）
uv sync --all-groups
```

### Pre-commit Hooks

本專案使用 pre-commit 確保程式碼品質，在每次 commit 前自動執行 linting 和 type checking。

```bash
# 安裝 pre-commit hooks
uv run pre-commit install

# 手動執行所有 hooks
uv run pre-commit run --all-files
```

Pre-commit 會自動執行以下檢查:

- **Ruff**: Python linter 和 code formatter
- **Mypy**: 靜態型別檢查
- 其他基本檢查（trailing whitespace、file endings 等）

### Code Quality Tools

#### Ruff

Ruff 是一個快速的 Python linter 和 formatter。

```bash
# 執行 linting
uv run ruff check .

# 自動修復問題
uv run ruff check --fix .

# 執行 code formatting
uv run ruff format .
```

#### Mypy

Mypy 用於靜態型別檢查。

```bash
# 執行型別檢查
uv run mypy behavior_analysis
```

### CI/CD

本專案使用 GitHub Actions 進行持續整合，在每次 PR 時會自動執行:

- Ruff linting
- Ruff formatting check
- Mypy type checking
- Pre-commit hooks

請確保本地通過所有檢查後再提交 PR。
