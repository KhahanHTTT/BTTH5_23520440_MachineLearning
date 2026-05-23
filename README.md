# K-Means Clustering
## Assignment 1
Thuật toán K-Means đã phân cụm thành công bộ dữ liệu thành 3 cụm riêng biệt tương ứng với các phân phối Gaussian ban đầu.

Các centroid hội tụ sau 6 vòng lặp, cho thấy quá trình tối ưu EM hoạt động hiệu quả và thuật toán hội tụ khá nhanh.

Kết quả trực quan cho thấy các cụm được phân tách rõ ràng và centroid nằm gần trung tâm của từng cụm dữ liệu.

Do centroid được khởi tạo ngẫu nhiên nên ở những lần chạy khác nhau, kết quả phân cụm hoặc số vòng lặp hội tụ có thể thay đổi nhẹ. Tuy nhiên trong lần chạy này, K-Means cho kết quả tốt và ổn định.


## Assignment 2
Trong Assignment 2, bộ dữ liệu gồm ba cụm Gaussian nhưng số lượng điểm dữ liệu giữa các cụm không đồng đều. 
Cụ thể, hai cụm đầu chỉ có 200 điểm trong khi cụm còn lại chứa tới 1000 điểm.

Kết quả trực quan cho thấy K-Means vẫn phân cụm thành công thành 3 cụm riêng biệt và các centroid nằm gần trung tâm của từng cụm dữ liệu. 
Thuật toán hội tụ sau 10 vòng lặp, cho thấy quá trình tối ưu EM hoạt động ổn định.

Tuy nhiên, so với Assignment 1, cụm có 1000 điểm tạo ảnh hưởng mạnh hơn đến quá trình cập nhật centroid. 
Do K-Means sử dụng giá trị trung bình để cập nhật tâm cụm, cụm lớn sẽ có tác động lớn hơn đến hàm mục tiêu của thuật toán.

Điều này cho thấy K-Means khá nhạy với dữ liệu mất cân bằng. 
Trong những trường hợp dữ liệu chồng lắp hoặc mất cân bằng nghiêm trọng hơn, các cụm nhỏ có thể bị gộp sai hoặc centroid bị lệch về phía cụm lớn.

Mặc dù vậy, trong thí nghiệm hiện tại, do khoảng cách giữa các cụm đủ xa nên K-Means vẫn đạt được kết quả phân cụm tốt và ổn định.

## Assignment 3
Trong Assignment 3, bộ dữ liệu gồm ba cụm Gaussian với số lượng điểm bằng nhau. 
Tuy nhiên, cụm thứ ba sử dụng ma trận hiệp phương sai:

Σ₂ = [[10, 0],
      [0, 1]]

Điều này làm cụm dữ liệu bị kéo dài theo trục x thay vì có dạng hình tròn như các cụm còn lại.

Kết quả trực quan cho thấy K-Means vẫn phân cụm được dữ liệu thành 3 cụm riêng biệt. 
Tuy nhiên, cụm bên phải có hình dạng trải dài và phân bố không đều hơn so với Assignment 1.

Do K-Means sử dụng khoảng cách Euclidean và giả định các cụm có dạng hình cầu, nên thuật toán gặp khó khăn hơn khi xử lý cụm bị kéo dài. 
Điều này làm số vòng lặp hội tụ tăng lên đáng kể.

Cụ thể:
- Assignment 1 hội tụ sau 6 vòng lặp
- Assignment 2 hội tụ sau 10 vòng lặp
- Assignment 3 hội tụ sau 19 vòng lặp

Điều đó cho thấy phân phối Gaussian bị kéo dài đã ảnh hưởng rõ rệt đến hiệu suất của K-Means.

Mặc dù vậy, do khoảng cách giữa các cụm vẫn khá xa nên thuật toán vẫn phân cụm đúng và các centroid vẫn nằm gần trung tâm của từng cụm dữ liệu.

## GMM Assignment 1
Kết quả trực quan cho thấy dữ liệu được phân cụm thành công thành 3 cụm tương ứng với các phân phối Gaussian ban đầu. 
Các mean của GMM nằm gần trung tâm của từng cụm dữ liệu.

So với K-Means, GMM sử dụng soft clustering nên mỗi điểm dữ liệu được gán xác suất thuộc về từng cụm thay vì chỉ thuộc duy nhất một cụm. 
Điều này giúp mô hình linh hoạt hơn khi xử lý dữ liệu Gaussian.

Thuật toán hội tụ sau 69 vòng lặp, nhiều hơn đáng kể so với K-Means. 
Nguyên nhân là do GMM cần đồng thời cập nhật mean, covariance và mixing weights trong mỗi vòng lặp EM.

## GMM Assignment 2
Trong Assignment 2, Gaussian Mixture Model được sử dụng để phân đoạn ảnh và tách nền của ảnh con bò.

Kết quả cho thấy GMM đã phân chia các pixel thành nhiều cụm màu khác nhau dựa trên phân phối xác suất Gaussian. 
Các vùng có màu sắc tương tự như nền cỏ, vùng đen của con bò và vùng sáng được gom thành các cụm riêng biệt.

Ảnh sau segmentation có số lượng màu giảm đáng kể, cho thấy GMM đã thực hiện clustering trên không gian màu của ảnh thành công.

So với K-Means, GMM hoạt động linh hoạt hơn do sử dụng mô hình xác suất và soft clustering, giúp mô hình hóa phân bố màu tốt hơn.

Kết quả cuối cùng cho thấy nền và đối tượng chính trong ảnh đã được phân tách tương đối rõ ràng