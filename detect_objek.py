import pygame # Digunakan untuk memutar suara alarm
import cv2 #Untuk menangani video atau input dari kamera, serta menampilkan hasil deteksi.
from ultralytics import YOLO #Memuat model YOLO untuk mendeteksi objek pada gambar atau video.

# Inisialisasi pygame untuk memutar suara
pygame.mixer.init()

# Fungsi untuk memutar alarm
def play_alarm(): # Memutar file audio alarm hanya jika alarm belum aktif. File suara disimpan di C:\\kebakaran\\alrm.mp3.
    if not pygame.mixer.music.get_busy():  # Periksa jika alarm belum berbunyi
        pygame.mixer.music.load("C:\\kebakaran\\alrm.mp3")  # Ganti dengan path file suara alarm Anda
        pygame.mixer.music.play(-1)  # Play berulang kali

# Fungsi untuk menghentikan alarm
def stop_alarm(): # Menghentikan suara alarm jika sedang berbunyi/Kode ini memastikan alarm tidak diputar berulang-ulang jika sudah berbunyi.
    if pygame.mixer.music.get_busy():  # Periksa jika alarm sedang berbunyi
        pygame.mixer.music.stop() # Menghentikan audio

# Load model YOLOv11
model = YOLO("C:\\kebakaran\\best.pt")  # Memuat model YOLOv11 yang telah dilatih (best.pt) untuk mendeteksi objek api/asap,Model ini adalah inti dari proses deteksi objek.

# Inisialisasi kamera
cap = cv2.VideoCapture(0)  # Menghubungkan ke kamera perangkat (default kamera utama).

if not cap.isOpened():#Jika kamera tidak berhasil diakses, program menampilkan pesan error dan berhenti.
    print("Kamera tidak dapat diakses!")
    exit()

while True:
    ret, frame = cap.read()#Mengambil frame dari kamera.
    if not ret:#Jika gagal membaca frame, loop dihentikan.
        print("Gagal membaca frame dari kamera!")
        break

    # Deteksi objek
    results = model(frame)#Menggunakan YOLO untuk menganalisis frame kamera dan mencari objek yang terdeteksi.

    # Jika objek ditemukan
    if results[0].boxes and len(results[0].boxes) > 0:#Ambil kotak deteksi (boxes) dan nama objek (names)
        boxes = results[0].boxes  # Akses kotak deteksi
        names = results[0].names  # Akses nama objek yang terdeteksi

        # Mengambil nama objek berdasarkan ID kelas
        detected_objects = [names[int(cls)] for cls in boxes.cls]  #Tampilkan nama objek di terminal.
        print(f"Objek terdeteksi: {detected_objects}")
        play_alarm()  # Nyalakan alarm menggunakan
    else:
        stop_alarm()  # Hentikan alarm jika tidak ada objek yang terdeteksi

    # Render hasil deteksi ke frame/Menggambar kotak dan label deteksi di atas frame asli.
    annotated_frame = results[0].plot()#

    # menampilkan hasil deteksi dalam jendela.
    cv2.imshow("Deteksi Objek", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) == ord('q'):
        break

# Bersihkan dan keluar
stop_alarm()#Memastikan suara alarm berhenti sebelum keluar.
cap.release()#Membebaskan kamera.
cv2.destroyAllWindows()#Menutup semua jendela yang dibuat OpenCV
