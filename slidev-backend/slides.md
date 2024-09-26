---
# You can also start simply with 'default'
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://cover.sli.dev
# some information about your slides (markdown enabled)
title: Welcome to Slidev
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
---

# Welcome to Slidev

Frontend ke Backend ke Awan?

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next page <carbon:arrow-right class="inline"/>
  </span>
</div>

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.com/slidevjs/slidev" target="_blank" alt="GitHub" title="Open in GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<!--
Bayangkan jika kita mengelola ribuan dokumen secara manual di dalam lemari arsip. Setiap kali kita ingin mencari satu dokumen tertentu, kita harus membuka lemari, memeriksa setiap folder, dan menyortir dokumen secara manual—tentu saja ini akan memakan waktu dan sangat tidak efisien. Nah, dalam dunia teknologi, lemari arsip itu bisa diibaratkan sebagai database (DB), dan teknologi memungkinkan kita untuk mengakses, mengelola, dan memperbarui data dengan cara yang jauh lebih cepat dan terstruktur.
-->

---
transition: fade-out
---

<img
  v-click
  class="absolute -bottom-9 -left-7 w-80 opacity-50"
  src="database.webp"
  alt=""
/>

<!--Database (DB) adalah bagian penting dalam pengembangan perangkat lunak karena memungkinkan kita untuk menyimpan dan mengelola data secara efisien. Database adalah kumpulan data yang terorganisir dengan baik, yang dapat diakses, dikelola, dan diperbarui dengan mudah. DB menyediakan cara untuk menyimpan data dalam format yang terstruktur dan terstandarisasi, sehingga mempermudah proses pengambilan, pengubahan, dan manipulasi data. -->

---
transition: slide-up
level: 2
---

<img
  v-click
  class="absolute -bottom-9 -left-7 w-80 opacity-50"
  src="database-table.png"
  alt=""
/>

<!--Karena database berperan penting dalam menyimpan dan mengelola informasi, kita membutuhkan cara untuk menghubungkannya dengan aplikasi yang digunakan oleh pengguna. Tidak mungkin aplikasi frontend, yang berinteraksi langsung dengan pengguna, berkomunikasi langsung dengan database. Itulah mengapa kita membutuhkan backend sebagai jembatan antara frontend dan database. Backend ini bertanggung jawab untuk memastikan bahwa data dari database bisa diambil, diproses, dan dikirimkan ke frontend agar pengguna bisa melihat atau menggunakannya.-->

---
layout: two-cols
layoutClass: gap-16
---

<img
  v-click
  class="absolute -bottom-9 -left-7 w-80 opacity-50"
  src="fe-be.png"
  alt=""
/>


<!--Bayangkan kamu sedang memesan makanan di restoran cepat saji. Ketika kamu memberikan pesananmu di kasir (frontend), kasir akan meneruskan pesanan tersebut ke dapur (backend). Di dapur, pesananmu disiapkan berdasarkan bahan-bahan yang tersedia (data di database). Setelah itu, kasir akan memberitahumu bahwa pesananmu siap. Begitu juga dengan cara aplikasi bekerja! -->

---
layout: image-right
image: https://cover.sli.dev
---

<!-- Frontend (Aplikasi di Depan Layar):
Ini adalah bagian dari aplikasi yang kamu lihat dan gunakan di layar ponsel atau komputer, misalnya aplikasi media sosial, toko online, atau aplikasi pesan instan. Frontend bertanggung jawab menampilkan informasi dan menerima masukan dari pengguna, seperti ketika kamu menekan tombol, mengisi formulir, atau mencari produk.

Backend (Aplikasi di Belakang Layar):
Frontend membutuhkan backend untuk melakukan tugas yang lebih rumit, seperti mengelola data atau mengirim informasi ke server. Backend ini bisa diibaratkan seperti dapur di restoran tadi, tempat di mana semua proses pengolahan terjadi di balik layar. Ketika kamu mengisi formulir di aplikasi (misalnya, mencari produk), frontend akan mengirimkan permintaan ke backend untuk memproses dan mendapatkan jawaban.-->



---
class: px-20
---



Cara Kerja Hubungannya:
Ketika kamu, sebagai pengguna, menggunakan aplikasi:

    Kamu membuat permintaan dari frontend—misalnya, kamu ingin melihat produk terbaru di toko online.
    Frontend mengirimkan permintaan tersebut ke backend melalui internet.
    Backend menerima permintaan, lalu mengambil data dari database—misalnya, daftar produk yang baru saja diunggah.
    Backend kemudian mengirimkan data yang telah diproses tadi kembali ke frontend, yang kemudian menampilkannya di layar ponsel atau komputermu

--- #9
class: px-20
---

<img
  v-click
  class="absolute -bottom-9 -left-7 w-80 opacity-50"
  src="supabase.jpg"
  alt=""
/>

Illustrasi Awan + Database

--- #9
class: px-20
---

Tunjukkin supabase

--- #11
class: px-20
---

Praktek dari FE ke supabase langsung, jelasin kalau supabase itu backend dan database jadi 1.


--- #12
class: px-20
---

Tambah realtime di db.
