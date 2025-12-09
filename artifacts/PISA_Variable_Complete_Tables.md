# PISA 2022 變數完整對應表

**包含原始問題與中文翻譯**


---


## 1️⃣ Access to Resources (資源可及性)


| 變數名稱 | 來源 | 原始問題 | 中文翻譯 | 回答選項 |
|---------|------|----------|----------|----------|
| **HOMEPOS** | Student | Home possessions (WLE) | 家庭財產指數（加權似然估計） | 連續變數 |
| **WORKHOME** | Student | Working in household/take care of family members before or a... | 上學前後在家做家務或照顧家人的頻率 | 0=不做家務或照顧家人, 1=每週做家務或照顧家人1次, 2=每週做家務或照顧家人2次, 3=每週做家務或照顧家人3次, 4=每週做家務或照顧家人4次, 5=每... |
| **SC017Q01NA** | School | Instruction hindered by: A lack of teaching staff | 教學受阻因素：缺乏教學人員 | 1=完全沒有, 2=很少, 3=某種程度, 4=很多 |
| **SC017Q02NA** | School | Instruction hindered by: Inadequate or poorly qualified teac... | 教學受阻因素：教學人員不足或素質不佳 | 1=完全沒有, 2=很少, 3=某種程度, 4=很多 |
| **SC017Q03NA** | School | Instruction hindered by: A lack of assisting staff | 教學受阻因素：缺乏輔助人員 | 1=完全沒有, 2=很少, 3=某種程度, 4=很多 |
| **SC017Q05NA** | School | Instruction hindered by: A lack of educational material (e.g... | 教學受阻因素：缺乏教材（如教科書、IT設備、圖書館或實驗室材料） | 1=完全沒有, 2=很少, 3=某種程度, 4=很多 |


## 2️⃣ Internet Access (網路與數位資源)


| 變數名稱 | 來源 | 原始問題 | 中文翻譯 | 回答選項 |
|---------|------|----------|----------|----------|
| **ICTRES** | Student | ICT Resources (WLE) | 資訊與通訊科技資源指數（加權似然估計） | 連續變數 |
| **ICTHOME** | Student | ICT availability outside of school  (WLE) | 校外資訊與通訊科技可用性（加權似然估計） | 連續變數 |
| **ICTAVHOM** | Student | Availability and Usage of ICT at Home | 家中資訊與通訊科技的可用性與使用情況 | 0=No ICT resources available at home, 1=1 ICT resource available at home, 2=2 IC... |


## 3️⃣ Learning Difficulties (學習困難)


| 變數名稱 | 來源 | 原始問題 | 中文翻譯 | 回答選項 |
|---------|------|----------|----------|----------|
| **ST127Q01TA** | Student | Have you ever repeated a [grade]: At [ISCED 1] | 你是否曾經留級：小學階段（ISCED 1） | 1=否，從未, 2=是，一次, 3=是，兩次或以上 |
| **ST127Q02TA** | Student | Have you ever repeated a [grade]: At [ISCED 2] | 你是否曾經留級：國中階段（ISCED 2） | 1=否，從未, 2=是，一次, 3=是，兩次或以上 |
| **ST127Q03TA** | Student | Have you ever repeated a [grade]: At [ISCED 3] | 你是否曾經留級：高中階段（ISCED 3） | 1=否，從未, 2=是，一次, 3=是，兩次或以上 |
| **ANXMAT** | Student | Mathematics Anxiety (WLE) | 數學焦慮指數（加權似然估計） | 連續變數 |


## 4️⃣ Geographic Disadvantage (地理劣勢)


| 變數名稱 | 來源 | 原始問題 | 中文翻譯 | 回答選項 |
|---------|------|----------|----------|----------|
| **SC001Q01TA** | School | Which of the following definitions best describes the commun... | 以下哪個定義最能描述你學校所在的社區？ | 1=村莊、小村或鄉村地區（少於3000人）, 2=小鎮（3000至約15000人）, 3=城鎮（15000至約100000人）, 4=城市（100000至約10... |


---


## 📝 詳細變數說明


### 1. Access to Resources (資源可及性)


#### `HOMEPOS`


**來源**: Student Questionnaire


**原始問題**: Home possessions (WLE)


**中文翻譯**: 家庭財產指數（加權似然估計）


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `WORKHOME`


**來源**: Student Questionnaire


**原始問題**: Working in household/take care of family members before or after school


**中文翻譯**: 上學前後在家做家務或照顧家人的頻率


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 0.0 | No work in household or care of family members | 不做家務或照顧家人 |

| 1.0 | 1 time of working in household or caring for family members per week | 每週做家務或照顧家人1次 |

| 2.0 | 2 times of working in household or caring for family members per week | 每週做家務或照顧家人2次 |

| 3.0 | 3 times of working in household or caring for family members per week | 每週做家務或照顧家人3次 |

| 4.0 | 4 times of working in household or caring for family members per week | 每週做家務或照顧家人4次 |

| 5.0 | 5 times of working in household or caring for family members per week | 每週做家務或照顧家人5次 |

| 6.0 | 6 times of working in household or caring for family members per week | 每週做家務或照顧家人6次 |

| 7.0 | 7 times of working in household or caring for family members per week | 每週做家務或照顧家人7次 |

| 8.0 | 8 times of working in household or caring for family members per week | 每週做家務或照顧家人8次 |

| 9.0 | 9 times of working in household or caring for family members per week | 每週做家務或照顧家人9次 |

| 10.0 | 10 or more times of working in household or caring for family members per week | 每週做家務或照顧家人10次 |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `SC017Q01NA`


**來源**: School Questionnaire


**原始問題**: Instruction hindered by: A lack of teaching staff


**中文翻譯**: 教學受阻因素：缺乏教學人員


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 1.0 | Not at all | 完全沒有 |

| 2.0 | Very little | 很少 |

| 3.0 | To some extent | 某種程度 |

| 4.0 | A lot | 很多 |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `SC017Q02NA`


**來源**: School Questionnaire


**原始問題**: Instruction hindered by: Inadequate or poorly qualified teaching staff


**中文翻譯**: 教學受阻因素：教學人員不足或素質不佳


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 1.0 | Not at all | 完全沒有 |

| 2.0 | Very little | 很少 |

| 3.0 | To some extent | 某種程度 |

| 4.0 | A lot | 很多 |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `SC017Q03NA`


**來源**: School Questionnaire


**原始問題**: Instruction hindered by: A lack of assisting staff


**中文翻譯**: 教學受阻因素：缺乏輔助人員


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 1.0 | Not at all | 完全沒有 |

| 2.0 | Very little | 很少 |

| 3.0 | To some extent | 某種程度 |

| 4.0 | A lot | 很多 |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `SC017Q05NA`


**來源**: School Questionnaire


**原始問題**: Instruction hindered by: A lack of educational material (e.g. textbooks, IT equipment, library or laboratory material)


**中文翻譯**: 教學受阻因素：缺乏教材（如教科書、IT設備、圖書館或實驗室材料）


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 1.0 | Not at all | 完全沒有 |

| 2.0 | Very little | 很少 |

| 3.0 | To some extent | 某種程度 |

| 4.0 | A lot | 很多 |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


### 2. Internet Access (網路與數位資源)


#### `ICTRES`


**來源**: Student Questionnaire


**原始問題**: ICT Resources (WLE)


**中文翻譯**: 資訊與通訊科技資源指數（加權似然估計）


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `ICTHOME`


**來源**: Student Questionnaire


**原始問題**: ICT availability outside of school  (WLE)


**中文翻譯**: 校外資訊與通訊科技可用性（加權似然估計）


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `ICTAVHOM`


**來源**: Student Questionnaire


**原始問題**: Availability and Usage of ICT at Home


**中文翻譯**: 家中資訊與通訊科技的可用性與使用情況


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 0.0 | No ICT resources available at home | No ICT resources available at home |

| 1.0 | 1 ICT resource available at home | 1 ICT resource available at home |

| 2.0 | 2 ICT resources available at home | 2 ICT resources available at home |

| 3.0 | 3 ICT resources available at home | 3 ICT resources available at home |

| 4.0 | 4 ICT resources available at home | 4 ICT resources available at home |

| 5.0 | 5 ICT resources available at home | 5 ICT resources available at home |

| 6.0 | 6 ICT resources available at home | 6 ICT resources available at home |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


### 3. Learning Difficulties (學習困難)


#### `ST127Q01TA`


**來源**: Student Questionnaire


**原始問題**: Have you ever repeated a [grade]: At [ISCED 1]


**中文翻譯**: 你是否曾經留級：小學階段（ISCED 1）


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 1.0 | No, never | 否，從未 |

| 2.0 | Yes, once | 是，一次 |

| 3.0 | Yes, twice or more | 是，兩次或以上 |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `ST127Q02TA`


**來源**: Student Questionnaire


**原始問題**: Have you ever repeated a [grade]: At [ISCED 2]


**中文翻譯**: 你是否曾經留級：國中階段（ISCED 2）


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 1.0 | No, never | 否，從未 |

| 2.0 | Yes, once | 是，一次 |

| 3.0 | Yes, twice or more | 是，兩次或以上 |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `ST127Q03TA`


**來源**: Student Questionnaire


**原始問題**: Have you ever repeated a [grade]: At [ISCED 3]


**中文翻譯**: 你是否曾經留級：高中階段（ISCED 3）


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 1.0 | No, never | 否，從未 |

| 2.0 | Yes, once | 是，一次 |

| 3.0 | Yes, twice or more | 是，兩次或以上 |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `ANXMAT`


**來源**: Student Questionnaire


**原始問題**: Mathematics Anxiety (WLE)


**中文翻譯**: 數學焦慮指數（加權似然估計）


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


### 4. Geographic Disadvantage (地理劣勢)


#### `SC001Q01TA`


**來源**: School Questionnaire


**原始問題**: Which of the following definitions best describes the community in which your school is located?


**中文翻譯**: 以下哪個定義最能描述你學校所在的社區？


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 1.0 | A village, hamlet or rural area (fewer than 3 000 people) | 村莊、小村或鄉村地區（少於3000人） |

| 2.0 | A small town (3 000 to about 15 000 people) | 小鎮（3000至約15000人） |

| 3.0 | A town (15 000 to about 100 000 people) | 城鎮（15000至約100000人） |

| 4.0 | A city (100 000 to about 1 000 000 people) | 城市（100000至約100萬人） |

| 5.0 | A large city (1 000 000 to about 10 000 000 people) | 大城市（100萬至約1000萬人） |

| 6.0 | A megacity (with over 10 000 000 people) | 超大城市（超過1000萬人） |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `ST260Q01JA`


**來源**: Student Questionnaire


**原始問題**: Have you ever missed school for more than three months in a row: At [ISCED 1]


**中文翻譯**: 你是否曾經連續缺課超過三個月：小學階段（ISCED 1）


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 1.0 | No, never | 否，從未 |

| 2.0 | Yes, once | 是，一次 |

| 3.0 | Yes, twice or more | 是，兩次或以上 |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `ST260Q02JA`


**來源**: Student Questionnaire


**原始問題**: Have you ever missed school for more than three months in a row: At [ISCED 2]


**中文翻譯**: 你是否曾經連續缺課超過三個月：國中階段（ISCED 2）


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 1.0 | No, never | 否，從未 |

| 2.0 | Yes, once | 是，一次 |

| 3.0 | Yes, twice or more | 是，兩次或以上 |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---


#### `ST260Q03JA`


**來源**: Student Questionnaire


**原始問題**: Have you ever missed school for more than three months in a row: At [ISCED 3]


**中文翻譯**: 你是否曾經連續缺課超過三個月：高中階段（ISCED 3）


**資料類型**: float64


**回答選項**:


| 代碼 | 原始標籤 | 中文翻譯 |

|------|----------|----------|

| 1.0 | No, never | 否，從未 |

| 2.0 | Yes, once | 是，一次 |

| 3.0 | Yes, twice or more | 是，兩次或以上 |

| 95.0 | Valid Skip | 有效跳題 |

| 97.0 | Not Applicable | 不適用 |

| 98.0 | Invalid | 無效 |

| 99.0 | No Response | 未回答 |



---
