"""Đo nhận dạng tai trên ảnh nhỏ thật (việc 2 của docs/story-imavis.md).

Ảnh nhỏ thật không có đáp án độ phân giải cao, nên PSNR không tính được. Nhãn danh tính thì có: phóng ảnh nhỏ bằng
từng phương pháp, rồi so đặc trưng với ảnh lớn của cùng người. Mạng nhận dạng đóng băng và chỉ học trên ảnh lớn của
những người không thuộc nhóm chấm.
"""
