# Lab 16 – Báo cáo benchmark LightGBM trên AWS (CPU)

**Môi trường:** AWS us-east-1, compute node `t3.micro` (2 vCPU, 913 MB RAM, thêm swap 2 GB), Ubuntu 22.04, LightGBM 4.7.0. Hạ tầng dựng bằng Terraform (VPC, Bastion, NAT Gateway, ALB, compute node ở private subnet).
**Dữ liệu:** Credit Card Fraud Detection (284.807 giao dịch, 492 gian lận ≈ 0,17%), chia train/test 80/20 có stratify.

## Kết quả

| Metric | Kết quả |
|---|---|
| Thời gian load data | 2,50 s |
| Thời gian training | 7,60 s |
| Best iteration | 68 |
| AUC-ROC | 0,9637 |
| Accuracy | 0,9995 |
| F1-Score | 0,8492 |
| Precision | 0,9383 |
| Recall | 0,7755 |
| Inference latency (1 dòng) | 1,68 ms |
| Inference throughput (1000 dòng) | ~266.000 dòng/s (3,8 ms / 1000 dòng) |

## Nhận xét

Mô hình LightGBM huấn luyện trên CPU `t3.micro` mất khoảng 7,6 giây (68 vòng boosting) sau khi load 284.807 dòng trong 2,5 giây, cho AUC-ROC 0,964, Precision 0,94, Recall 0,78 và F1 0,85. Accuracy 99,95% không có nhiều ý nghĩa vì lớp gian lận chỉ chiếm 0,17%, nên cần xem Precision, Recall, AUC. Inference một dòng mất khoảng 1,7 ms và dự đoán theo lô 1000 dòng chỉ mất vài mili-giây, đủ nhanh cho dữ liệu dạng bảng mà không cần GPU. Khi huấn luyện, CPU đạt đỉnh khoảng 18% và RAM dùng khoảng 190 MB (EC2 Monitoring và `free -h`), nên `t3.micro` đủ cho bài này, nhưng chỉ có 1 GB RAM nên tôi thêm swap để tránh OOM.

Lưu ý khi chạy: với cấu hình mặc định, mô hình bị bất ổn do mất cân bằng lớp nặng (AUC chỉ 0,93 và dừng ở vòng 1). Tôi đã thêm `min_sum_hessian_in_leaf=1.0`, `reg_lambda=1.0`, `metric="auc"` và tăng patience early stopping lên 150 để mô hình ổn định. Kết quả cũng nhạy với tập validation nhỏ (chỉ 39 mẫu gian lận), nên các chỉ số có thể dao động giữa các lần chia dữ liệu.

## Chi phí

Billing Dashboard có độ trễ khoảng 24 giờ nên ngày chạy lab chưa hiện chi phí; ngày hôm sau, trang Bills (kỳ 1–31/10/2026) ghi nhận khoảng **$0,99** chi phí phát sinh tại us-east-1 và trang Credits ghi **$0,98** ước tính đã dùng, còn lại $119,02 trên tổng $120 credit (AWS Free Tier $100 và Explore AWS $20). Do đó tổng thanh toán thực tế là $0,00 vì credit chi trả toàn bộ. Số này khớp với ước tính theo giờ (us-east-1): 2 EC2 `t3.micro` ~$0,021, NAT Gateway ~$0,045 (chưa tính dữ liệu), ALB ~$0,02, tổng khoảng $0,09/giờ trong vài giờ chạy. Ảnh EC2 Instances, NAT Gateway và Load Balancer cho thấy các dịch vụ phát sinh phí. Hạ tầng đã được xóa bằng `terraform destroy` (27 resource) sau khi chụp xong ảnh.

## Danh sách file nộp

- `screenshots/`: output benchmark, `free -h`/`top`, EC2 Monitoring, EC2 Instances, NAT Gateway, Load Balancer, `terraform destroy`, AWS Billing (Bills) và Credits
- `benchmark_result.json`, `benchmark.py`
- `terraform/` (đã loại private key, state và thư mục `.terraform`)
