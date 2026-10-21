# Hướng dẫn làm phần Data trong dự án MLOps (từng bước)

> Mục đích: sau này vào bất kỳ dự án ML nào và nhận phần **Data**, bạn biết **làm gì, theo thứ tự nào, và vì sao**.
> Ví dụ xuyên suốt: CSE-CIC-IDS2018 (phát hiện xâm nhập mạng).
>
> **Nguồn dữ liệu:** dùng bộ đầy đủ 10 file từ nguồn chính thức của CIC (xem Bước 2), **không dùng** bản Kaggle `primus11` vì nó chỉ có một ngày.
>
> **Lưu ý:** tên cột và số liệu trong code là theo hiểu biết về CIC-IDS-2018 gốc. Bản Kaggle có thể khác, nên ở **Bước 3** luôn phải nhìn dữ liệu thật trước rồi mới sửa code cho khớp.

---

## Phần 0. Tư duy trước khi làm

**Phần Data trong MLOps khác gì "làm data cho bài tập ML"?**

| Bài tập ML | Dự án MLOps |
|---|---|
| Mở notebook, làm sạch, train, xong | Mọi bước là **code chạy lại được** bằng một lệnh |
| Sửa dữ liệu tay, quên đã sửa gì | Mỗi thay đổi được **ghi lại và có version** |
| Chia dữ liệu ngẫu nhiên | Chia **có chủ đích** (tránh rò rỉ, mô phỏng thực tế) |
| Chỉ phục vụ lúc train | Dữ liệu và bộ tiền xử lý còn dùng lại ở **API** và **giám sát Drift** |

Người làm Data phải trả lời được 5 câu hỏi:
1. Dữ liệu **đến từ đâu**, phiên bản nào?
2. Dữ liệu **có đáng tin** không (sạch, đúng schema)?
3. Đã **biến đổi** thế nào để ra bộ dữ liệu huấn luyện?
4. Chia tập **có bị rò rỉ** không?
5. Người khác chạy lại có ra **y hệt** không?

Phần còn lại của hướng dẫn là cách trả lời 5 câu này.

---

## Bước 1. Dựng môi trường và khung thư mục

**Vì sao:** môi trường riêng giúp mọi người dùng cùng phiên bản thư viện. Khung thư mục rõ ràng giúp tách *dữ liệu gốc* khỏi *dữ liệu đã xử lý*.

```bash
cd CIC-IDS-MLOps
python3 -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow matplotlib seaborn pandera pyyaml scikit-learn kaggle jupyter pytest
pip freeze > requirements.txt   # chốt phiên bản
```

Khung thư mục (đã có sẵn trong dự án):

```text
data/raw/        # dữ liệu gốc, KHÔNG SỬA TAY, không commit
data/interim/    # bước trung gian (Parquet)
data/processed/  # train / val / test cuối cùng
reports/data/    # báo cáo, biểu đồ
src/data/        # code xử lý
tests/data/      # test
params.yaml      # mọi tham số (không viết cứng trong code)
```

**`.gitignore`** (dữ liệu lớn và bí mật không được đẩy lên Git):

```text
.venv/
data/raw/
data/interim/
data/processed/
kaggle.json
```

> **Quy tắc vàng:** `data/raw` là **bất biến**. Sai ở đâu thì sửa **code** rồi chạy lại, không sửa file gốc.

---

## Bước 2. Lấy dữ liệu và ghi dấu vân tay (checksum)

**Vì sao:** để chứng minh "dữ liệu tôi dùng hôm nay đúng là dữ liệu hôm qua", và phát hiện file tải lỗi.

### 2.1. Chọn nguồn dữ liệu (bài học thực tế)

Bản trên Kaggle (`primus11/cic-ids-2018-dataset`) **chỉ có 1 file `cic.csv`** (342 MB). Kiểm tra bằng EDA cho thấy đó chỉ là **ngày 14/02/2018** của bộ gốc (1.048.575 dòng, 3 nhãn: Benign, FTP-BruteForce, SSH-Bruteforce). Thiếu DoS, DDoS, Botnet, Web Attack, Infiltration và không có nhiều ngày để chia theo thời gian.

> **Bài học:** tên dataset trên Kaggle không đảm bảo là bộ đầy đủ. Trước khi dùng, luôn đối chiếu với **nguồn gốc** (trang của nhóm tác giả) về số file, dung lượng, số nhãn và khoảng thời gian.

Bộ **đầy đủ** lấy từ nguồn chính thức của CIC:
- Trang gốc: <https://www.unb.ca/cic/datasets/ids-2018.html>
- AWS Open Data: <https://registry.opendata.aws/cse-cic-ids2018/> (bucket công khai `s3://cse-cic-ids2018`, vùng `ca-central-1`, **không cần tài khoản AWS**)
- Thư mục cần lấy: `Processed Traffic Data for ML Algorithms/` gồm **10 file CSV** (mỗi ngày một file), tổng **6,4 GiB**. Thư mục `Original Network Traffic and Log data/` là pcap thô, rất nặng, **không cần**.

| File (mỗi file một ngày) | Dung lượng |
|---|---|
| Wednesday-14-02-2018 | 342 MiB |
| Thursday-15-02-2018 | 359 MiB |
| Friday-16-02-2018 | 318 MiB |
| Thuesday-20-02-2018 *(tên gốc viết sai chính tả, giữ nguyên)* | **3,8 GiB** |
| Wednesday-21-02-2018 | 314 MiB |
| Thursday-22-02-2018 | 365 MiB |
| Friday-23-02-2018 | 365 MiB |
| Wednesday-28-02-2018 | 200 MiB |
| Thursday-01-03-2018 | 103 MiB |
| Friday-02-03-2018 | 336 MiB |

(Tên file đầy đủ có hậu tố `_TrafficForML_CICFlowMeter.csv`.)

### 2.2. Kiểm tra trước khi tải (không tải gì cả)

Luôn **liệt kê trước, tải sau** để biết dung lượng và so với ổ đĩa còn trống (`df -h .`):

```bash
python -m awscli s3 ls --no-sign-request --region ca-central-1 --human-readable --summarize \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/"
```

### 2.3. Tải về

Cài AWS CLI vào venv (`pip install awscli`) rồi tải thẳng vào `data/raw/`:

```bash
python -m awscli s3 sync --no-sign-request --region ca-central-1 \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" data/raw/
```

Giải thích: `sync` chỉ tải file chưa có hoặc đổi kích thước nên **bị ngắt thì chạy lại đúng lệnh cũ để tải tiếp**. `--no-sign-request` nghĩa là truy cập ẩn danh, không cần khóa AWS.

Lưu ý thực tế:
- **Dùng `python -m awscli` thay vì `aws`** nếu đường dẫn dự án có khoảng trắng hoặc dấu tiếng Việt (script `aws` trong venv sẽ lỗi `bad interpreter`).
- Tốc độ khoảng 2 MiB/s thì 6,4 GiB mất gần 1 giờ; không để máy ngủ.
- Trong lúc tải, file có đuôi ngẫu nhiên (ví dụ `.Eb637a00`) là file tạm, sẽ tự đổi tên khi xong.
- Nếu trước đó đã có `cic.csv` từ Kaggle thì file đó **trùng** với `Wednesday-14-02-2018...csv`: xóa một trong hai (sau khi so sánh checksum) để không bị trùng dữ liệu.
- Ổ đĩa phải còn trống thoải mái: 6,4 GiB dữ liệu thô + Parquet trung gian + bản đã xử lý.
- Có thể tải trực tiếp từng file bằng trình duyệt qua link dạng `https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/<tên file>`.

### 2.4. Ghi dấu vân tay (checksum)

Viết `src/data/download.py` để tạo `MANIFEST.json` (tên file → SHA-256 → kích thước):

```python
import hashlib, json
from pathlib import Path

RAW = Path("data/raw")

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()

manifest = {
    p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
    for p in sorted(RAW.glob("*.csv"))
}
Path("data/raw/MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(f"{len(manifest)} files")
```

**Kiểm tra:** `MANIFEST.json` có đủ **10 file**, dung lượng từng file khớp bảng ở mục 2.1 (tổng khoảng 6,4 GiB). Thiếu file hoặc lệch dung lượng nghĩa là tải chưa xong hoặc bị lỗi, hãy chạy lại lệnh `sync`.

---

## Bước 3. Nhìn dữ liệu trước khi làm bất cứ điều gì

**Vì sao:** đây là bước nhiều người bỏ qua nhất và gây lỗi nhiều nhất. Đừng giả định, hãy nhìn.

Mở Jupyter (`jupyter notebook`) và chạy thử trên **một file**:

```python
import pandas as pd
from pathlib import Path

files = sorted(Path("data/raw").glob("*.csv"))
for f in files:
    print(f.name, round(f.stat().st_size / 1e6), "MB")

df = pd.read_csv(files[0], nrows=5000)   # đọc mẫu nhỏ cho nhanh
print(df.shape)
df.head()
df.info()
```

Trả lời các câu hỏi sau và **ghi lại**:

| Câu hỏi | Cách xem |
|---|---|
| Có bao nhiêu file, mỗi file bao nhiêu dòng? | Đếm dòng từng file |
| Tên cột có giống nhau giữa các file không? | So sánh `set(df.columns)` giữa các file |
| Cột nhãn tên gì? Giá trị ra sao? | `df["Label"].value_counts()` |
| Có cột thời gian không, định dạng gì? | `df["Timestamp"].head()` |
| Có cột kiểu `object` mà đáng ra là số không? | `df.info()` |
| Có dòng header bị lặp trong dữ liệu không? | `(df["Label"] == "Label").sum()` |

So sánh cột giữa các file:

```python
cols = {f.name: set(pd.read_csv(f, nrows=1).columns) for f in files}
common = set.intersection(*cols.values())
for name, c in cols.items():
    print(name, "- cột thừa so với các file khác:", c - common)
print("số cột chung:", len(common))
```

> Nếu có file thừa/thiếu cột so với các file khác (trong CIC-IDS-2018 gốc có một file như vậy), đó là một **phát hiện cần ghi vào báo cáo** và xử lý ở Bước 6.

---

## Bước 4. EDA (phân tích khám phá), đây là phần "có biểu đồ"

**Vì sao:** EDA cho bạn **căn cứ** để quyết định cách làm sạch và cách chia tập. Mỗi biểu đồ phải trả lời một câu hỏi cụ thể, không vẽ cho đẹp.

Đầu tiên gộp dữ liệu (đọc từng file, ép kiểu số, downcast để tiết kiệm RAM):

```python
def load_one(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df["Label"] != "Label"]            # bỏ dòng header lặp (nếu có)
    label = df["Label"]
    num = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce")
    num = num.apply(pd.to_numeric, downcast="float")
    num["Label"] = label
    num["source_file"] = path.name
    return num
```

> `errors="coerce"` biến giá trị không phải số thành `NaN`. Cột kiểu thời gian (`Timestamp`) sẽ thành `NaN` nếu ép số, nên xử lý riêng bằng `pd.to_datetime(..., errors="coerce")`. Đây là chỗ cần điều chỉnh theo schema thật.

### Biểu đồ 1: Phân bố nhãn
**Câu hỏi:** dữ liệu cân bằng không?

```python
import matplotlib.pyplot as plt
counts = df["Label"].value_counts()
ax = counts.plot.barh(logy=False)
ax.set_xscale("log")      # thang log vì chênh lệch rất lớn
plt.title("Phân bố nhãn (thang log)")
plt.tight_layout(); plt.savefig("reports/data/figures/label_dist.png", dpi=150)
```
**Rút ra:** nếu Benign áp đảo → không dùng Accuracy; dùng Recall, PR-AUC, FPR; cần xử lý mất cân bằng (chỉ ở tập train).

### Biểu đồ 2: Nhãn theo từng ngày (file)
**Câu hỏi:** loại tấn công nào xuất hiện ở ngày nào? Quyết định cách chia tập.

```python
pd.crosstab(df["source_file"], df["Label"]).plot.bar(stacked=True, logy=True, figsize=(12, 5))
plt.title("Số luồng theo ngày và nhãn")
plt.tight_layout(); plt.savefig("reports/data/figures/label_by_day.png", dpi=150)
```
**Rút ra:** nếu một loại tấn công chỉ có ở một ngày → chia tập theo ngày thuần túy sẽ làm train không thấy loại đó (xem Bước 10).

### Biểu đồ 3: Dữ liệu lỗi theo cột
**Câu hỏi:** cột nào có `NaN`, `inf`, giá trị âm?

```python
import numpy as np
X = df.drop(columns=["Label", "source_file"]).select_dtypes("number")
bad = pd.DataFrame({
    "nan": X.isna().mean(),
    "inf": np.isinf(X).mean(),
    "neg": (X < 0).mean(),
})
bad[bad.sum(axis=1) > 0].sort_values("nan", ascending=False).plot.bar(figsize=(12, 4))
plt.title("Tỉ lệ NaN / inf / giá trị âm theo cột")
plt.tight_layout(); plt.savefig("reports/data/figures/bad_values.png", dpi=150)
```
**Rút ra:** danh sách cột cần xử lý ở bước làm sạch.

### Biểu đồ 4: Trùng lặp và cột hằng số
```python
print("dòng trùng:", X.duplicated().sum(), f"({X.duplicated().mean():.2%})")
const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
print("cột hằng số:", const_cols)
```

### Biểu đồ 5: Tương quan giữa các đặc trưng
**Câu hỏi:** có đặc trưng nào gần như giống hệt nhau (thừa) không?

```python
import seaborn as sns
sample = X.sample(min(len(X), 200_000), random_state=42)   # lấy mẫu cho nhanh
corr = sample.corr()
plt.figure(figsize=(14, 12)); sns.heatmap(corr, cmap="coolwarm", center=0)
plt.tight_layout(); plt.savefig("reports/data/figures/corr.png", dpi=120)
```

### Biểu đồ 6: Benign so với Attack trên vài đặc trưng chính
```python
feat = ["Flow Duration", "Tot Fwd Pkts", "Pkt Len Mean"]   # đổi theo tên cột thật
for c in feat:
    sns.boxplot(data=sample.assign(y=df.loc[sample.index, "Label"] != "Benign"),
                x="y", y=c, showfliers=False)
    plt.yscale("symlog"); plt.title(c); plt.show()
```
**Rút ra:** đặc trưng nào phân tách tốt hai lớp.

> **Mẹo viết báo cáo:** dưới mỗi biểu đồ ghi 1 đến 2 câu: *"Quan sát → Hệ quả cho quyết định xử lý"*. Đó là điều làm phần Data có giá trị.

---

## Bước 5. Ghi quyết định vào `params.yaml`

**Vì sao:** các quyết định (cột nào bỏ, chia thế nào, seed bao nhiêu) phải nằm ở **một chỗ**, đọc được và thay đổi được mà không sửa code.

```yaml
data:
  raw_dir: data/raw
  interim_dir: data/interim
  processed_dir: data/processed
  label_col: Label
  benign_label: Benign
  time_col: Timestamp
  drop_cols: [Flow ID, Src IP, Dst IP, Src Port, Timestamp]   # sửa theo EDA
  seed: 42
split:
  strategy: by_day            # hoặc within_day_time
  train_days: []              # điền sau EDA
  val_days: []
  test_days: []
preprocess:
  scaler: robust
  imbalance: class_weight     # class_weight | undersample | smote
```

---

## Bước 6. Ingest: gộp, chuẩn hóa, lưu Parquet

**Vì sao Parquet?** Nhỏ hơn CSV nhiều lần, đọc nhanh, **nhớ kiểu dữ liệu** (CSV thì mỗi lần đọc lại phải đoán).

```python
# src/data/ingest.py
import pandas as pd, yaml
from pathlib import Path

P = yaml.safe_load(open("params.yaml"))["data"]
out = Path(P["interim_dir"]); out.mkdir(parents=True, exist_ok=True)

for f in sorted(Path(P["raw_dir"]).glob("*.csv")):
    df = load_one(f)                      # hàm ở Bước 4
    # xử lý Timestamp
    if P["time_col"] in df.columns:
        df[P["time_col"]] = pd.to_datetime(df[P["time_col"]], errors="coerce")
    df.to_parquet(out / f"{f.stem}.parquet", index=False)
    print(f.name, df.shape)
```

> Nếu ở Bước 3 bạn thấy file thừa/thiếu cột: thống nhất bằng cách giữ **tập cột chung**, hoặc thêm cột còn thiếu bằng `NaN`. Ghi quyết định này vào báo cáo.

---

## Bước 7. Validate: kiểm tra dữ liệu tự động

**Vì sao:** để pipeline **tự dừng** khi dữ liệu sai, thay vì lặng lẽ train trên dữ liệu hỏng. Đây là điểm đặc trưng của MLOps.

Cách đơn giản nhất (assert) rồi nâng cấp lên Pandera/Great Expectations:

```python
# src/data/validate.py
import numpy as np, pandas as pd

def validate(df, label_col="Label", allowed_labels=None, max_null=0.01):
    errors = []
    if label_col not in df.columns:
        errors.append("Thiếu cột nhãn")
    elif allowed_labels is not None:
        unknown = set(df[label_col].unique()) - set(allowed_labels)
        if unknown:
            errors.append(f"Nhãn lạ: {unknown}")
    num = df.select_dtypes("number")
    if np.isinf(num).any().any():
        errors.append("Còn giá trị inf")
    null_rate = num.isna().mean().max()
    if null_rate > max_null:
        errors.append(f"Tỉ lệ NaN cao nhất {null_rate:.2%} > {max_null:.0%}")
    if errors:
        raise ValueError("Validate thất bại:\n- " + "\n- ".join(errors))
```

Chạy validate **hai lần**: sau Ingest (để biết dữ liệu gốc lỗi thế nào, có thể chỉ cảnh báo) và sau Clean (phải đạt, nếu không thì dừng).

---

## Bước 8. Clean: làm sạch có thống kê

**Vì sao thống kê?** Mỗi lần xóa dòng là một quyết định ảnh hưởng đến mô hình, cần có số liệu trước và sau.

```python
# src/data/clean.py
import numpy as np, json

def clean(df, drop_cols, report):
    report["rows_in"] = len(df)

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df = df.replace([np.inf, -np.inf], np.nan)
    report["rows_with_nan_or_inf"] = int(df.isna().any(axis=1).sum())
    df = df.dropna()

    num_cols = df.select_dtypes("number").columns
    neg_mask = (df[num_cols] < 0).any(axis=1)
    report["rows_with_negative"] = int(neg_mask.sum())
    # Quyết định (loại hay giữ) phụ thuộc EDA: cột nào âm là "lỗi", cột nào âm hợp lệ?

    n = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = n - len(df)

    const = [c for c in num_cols if df[c].nunique() <= 1]
    df = df.drop(columns=const)
    report["constant_cols_dropped"] = const

    report["rows_out"] = len(df)
    return df
```

Ghi `report` ra `reports/data/cleaning_report.json`.

> **Cảnh báo về trùng lặp:** nếu bạn chia theo thời gian, nên loại trùng **trước khi chia** để cùng một bản ghi không rơi vào cả train lẫn test.

---

## Bước 9. Gán nhãn

```python
df["y_binary"] = (df["Label"] != "Benign").astype("int8")   # 0 = Benign, 1 = Attack
```

Với đa lớp: tạo `label_map = {tên_nhãn: số}` rồi lưu ra `artifacts/preprocessing/label_map.json`. Lớp quá hiếm (vài chục mẫu) → gộp hoặc loại, và ghi lại.

---

## Bước 10. Chia tập (train / val / test), bước dễ sai nhất

**Rò rỉ dữ liệu (data leakage)** là khi thông tin của tập test "lọt" vào lúc train, khiến kết quả đẹp giả tạo. Với dữ liệu mạng, chia ngẫu nhiên nghĩa là các luồng cùng một phiên tấn công nằm cả trong train và test → mô hình "nhớ" thay vì "hiểu".

Hai phương án (chọn dựa trên **Biểu đồ 2**):

| Phương án | Ưu | Nhược |
|---|---|---|
| **A. Chia theo ngày** | Sát thực tế, thể hiện rõ Drift | Có thể thiếu loại tấn công ở tập train |
| **B. Theo thời gian trong từng ngày** (đầu ngày → train, cuối ngày → test) | Mọi tập đều có đủ lớp | Ít thể hiện Drift hơn |

Ví dụ phương án B:

```python
df = df.sort_values("Timestamp")
parts = []
for day, g in df.groupby("source_file"):
    n = len(g); a, b = int(n * 0.70), int(n * 0.85)
    parts.append((g.iloc[:a], g.iloc[a:b], g.iloc[b:]))
train = pd.concat(p[0] for p in parts)
val   = pd.concat(p[1] for p in parts)
test  = pd.concat(p[2] for p in parts)
```

Kiểm tra bắt buộc sau khi chia:
- Ba tập **không giao nhau**.
- Tập nào cũng có đủ các lớp cần thiết (in `value_counts`).
- Ghi vào báo cáo kích thước và phân bố nhãn của từng tập.

Các ngày **không dùng để train** → lưu thành `production_sim/` để mô phỏng dữ liệu vận hành và Drift sau này.

---

## Bước 11. Tiền xử lý (học từ train, áp dụng cho tất cả)

**Nguyên tắc:** bất cứ thứ gì "học" từ dữ liệu (trung bình, độ lệch chuẩn, danh sách đặc trưng) chỉ được học từ **train**.

```python
from sklearn.preprocessing import RobustScaler
import joblib, json

features = [c for c in train.columns if c not in ("Label", "y_binary", "source_file")]
scaler = RobustScaler().fit(train[features])         # CHỈ fit trên train
train[features] = scaler.transform(train[features])
val[features]   = scaler.transform(val[features])    # val/test chỉ transform
test[features]  = scaler.transform(test[features])

joblib.dump(scaler, "artifacts/preprocessing/scaler.pkl")
json.dump(features, open("artifacts/preprocessing/features.json", "w"))
```

- **Mất cân bằng lớp:** ưu tiên `class_weight` / `scale_pos_weight` (đơn giản, không tạo dữ liệu giả). Nếu undersample/SMOTE thì **chỉ trên train**, giữ nguyên val/test.
- **Vì sao lưu `scaler.pkl`, `features.json`?** API khi nhận request mới phải biến đổi **y hệt** lúc train, nếu không sẽ bị *train–serve skew* (lệch giữa huấn luyện và phục vụ).

Lưu `train/val/test.parquet` vào `data/processed/`.

---

## Bước 12. Tái lập được: ghi lại mọi thứ

Tối thiểu mỗi lần huấn luyện phải log vào MLflow:
- `params.yaml` (các tham số dữ liệu)
- Hash của dữ liệu đầu vào (từ `MANIFEST.json`)
- Số dòng và phân bố nhãn của train/val/test
- Artifact: `scaler.pkl`, `features.json`, `label_map.json`

```python
import mlflow
with mlflow.start_run():
    mlflow.log_artifact("params.yaml")
    mlflow.log_artifact("data/raw/MANIFEST.json")
    mlflow.log_dict({"train": len(train), "val": len(val), "test": len(test)}, "data_sizes.json")
```

Nâng cao: dùng **DVC** để gắn version dữ liệu với commit Git (`dvc init`, `dvc add data/processed`, `dvc remote add ...`, `dvc push`).

---

## Bước 13. Viết test cho code dữ liệu

```python
# tests/data/test_split.py
def test_no_overlap(train, val, test):
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)

def test_time_order(train, val):
    assert train["Timestamp"].max() <= val["Timestamp"].max()
```

Test nên kiểm tra tối thiểu: không còn `inf`, nhãn hợp lệ, ba tập không giao nhau, scaler không fit trên val/test. Chạy `pytest tests/data`.

---

## Bước 14. Viết báo cáo / Data Card

Một báo cáo Data tốt có bố cục:
1. **Nguồn và phạm vi** của dữ liệu (kèm hash/phiên bản)
2. **EDA** (các biểu đồ ở Bước 4 + nhận xét ngắn)
3. **Vấn đề chất lượng phát hiện được** (kèm số liệu)
4. **Quy tắc làm sạch** và số dòng trước/sau
5. **Cách gán nhãn** và **cách chia tập** (kèm lý do)
6. **Tiền xử lý** và artifact tạo ra
7. **Giới hạn đã biết** (dữ liệu mô phỏng, mất cân bằng, v.v.)
8. **Cách chạy lại** (các lệnh)

---

## Phụ lục A. Lỗi người mới hay mắc

| Lỗi | Hậu quả | Cách tránh |
|---|---|---|
| Scale/chọn đặc trưng **trước** khi chia tập | Rò rỉ thông tin test | Chia trước, `fit` chỉ trên train |
| Chia ngẫu nhiên dữ liệu có yếu tố thời gian | Kết quả đẹp giả | Chia theo thời gian |
| Giữ IP, Flow ID làm đặc trưng | Mô hình học thuộc máy | Bỏ cột định danh |
| Dùng Accuracy khi dữ liệu mất cân bằng | 99% nhưng bỏ sót hết tấn công | Dùng Recall, PR-AUC, FPR |
| Cân bằng lớp trên cả val/test | Đánh giá sai thực tế | Chỉ cân bằng train |
| Sửa file trong `data/raw` | Không tái lập được | Raw bất biến, sửa bằng code |
| Hard-code tham số trong code | Khó thay đổi, khó ghi lại | Đưa vào `params.yaml` |
| Quên lưu scaler | API không tiền xử lý giống lúc train | Log artifact vào MLflow |

## Phụ lục B. Checklist cho **bất kỳ** dự án nào (dùng lại sau này)

- [ ] Dữ liệu đến từ đâu, phiên bản nào, giấy phép gì?
- [ ] Có `MANIFEST`/checksum hoặc DVC không?
- [ ] Đã **nhìn** dữ liệu thật (schema, kiểu, nhãn, thời gian) chưa?
- [ ] EDA trả lời được: cân bằng lớp, lỗi dữ liệu, trùng lặp, phân bố theo thời gian?
- [ ] Biết cột nào có thể gây **rò rỉ** (ID, thời gian, thông tin sinh ra sau nhãn)?
- [ ] Có bước validate tự động và pipeline dừng khi sai không?
- [ ] Chia tập có phản ánh cách mô hình sẽ dùng thực tế không?
- [ ] Mọi bước học tham số chỉ fit trên train?
- [ ] Artifact tiền xử lý đã được lưu để API dùng lại?
- [ ] Chạy lại từ đầu có ra kết quả giống hệt (cùng seed, cùng hash)?
- [ ] Có test và có tài liệu (Data Card)?
