import os
from dotenv import load_dotenv
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import joblib

# Muat environment variables dari file .env
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

dt_model = joblib.load(os.path.join(current_dir, 'dt_model.joblib'))
feature_columns = joblib.load(os.path.join(current_dir, 'feature_columns.joblib'))
genres = joblib.load(os.path.join(current_dir, 'genres.joblib'))
df_processed = joblib.load(os.path.join(current_dir, 'df_processed.joblib'))

# Menyimpan input pengguna sementara
user_inputs = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Memulai bot dan meminta pengguna untuk memilih genre."""
    keyboard = [[KeyboardButton(genre)] for genre in genres]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    await update.message.reply_text("Pilih genre favorit Anda:", reply_markup=reply_markup)

async def genre_selected(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menyimpan genre dan meminta tahun rilis sesuai dengan genre tersebut."""
    user_id = update.effective_user.id
    genre = update.message.text
    user_inputs[user_id] = {'genre': genre}
    
    # Filter dataset berdasarkan genre yang dipilih
    genre_col = f"Genre_{genre}"
    filtered_df = df_processed[df_processed[genre_col] == 1]
    years = filtered_df['Tahun Rilis'].unique()

    # Tampilkan pilihan tahun rilis
    keyboard = [[KeyboardButton(str(year))] for year in years]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    await update.message.reply_text(f"Pilih tahun rilis untuk genre {genre}:", reply_markup=reply_markup)

async def year_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menggunakan Decision Tree untuk memprediksi film yang sesuai dan menampilkannya."""
    user_id = update.effective_user.id
    try:
        year = int(update.message.text)
    except ValueError:
        await update.message.reply_text("Tahun rilis tidak valid. Silakan masukkan angka, misalnya 2020.")
        return

    user_inputs[user_id]['year'] = year
    genre = user_inputs[user_id]['genre']
    genre_col = f"Genre_{genre}"

    # Filter dataset berdasarkan genre dan tahun rilis
    filtered_df = df_processed[(df_processed[genre_col] == 1) & (df_processed['Tahun Rilis'] == year)]

    if filtered_df.empty:
        await update.message.reply_text("Maaf, tidak ada film yang sesuai dengan preferensi Anda.")
        return

    # Prediksi menggunakan Decision Tree
    X_filtered = filtered_df[feature_columns]
    predictions = dt_model.predict(X_filtered)

    # Ambil film yang sesuai dengan prediksi
    recommended_movies = filtered_df[filtered_df['Judul'].isin(predictions)]

    # Urutkan berdasarkan jumlah penonton terbanyak
    sorted_recommendations = recommended_movies.sort_values(by='Penonton', ascending=False)

    # Ambil hanya 3 film terbaik
    max_recommendations = 3
    recommendations = []
    for _, row in sorted_recommendations.head(max_recommendations).iterrows():
        jumlah_penonton = f"{row['Penonton']:,}".replace(",", ".")  # Format jumlah penonton
        movie_info = (
            f"🎬 *Judul:* {row['Judul']}\n"
            f"🎭 *Genre:* {genre}\n"
            f"📅 *Tahun Rilis:* {row['Tahun Rilis']}\n"
            f"🔞 *Klasifikasi Usia:* {row['Klasifikasi usia']}\n"
            f"🎭 *Pemeran:* {row['Pemeran']}\n"
            f"🎬 *Sutradara:* {row['Sutradara']}\n"
            f"🏢 *Produksi:* {row['Produksi']}\n"
            f"👥 *Jumlah Penonton:* {jumlah_penonton}"
        )
        recommendations.append(movie_info)

    # Kirimkan hasil rekomendasi
    for rec in recommendations:
        await update.message.reply_text(rec, parse_mode="Markdown")

    # Tampilkan opsi untuk rekomendasi lain atau selesai
    keyboard = [
        [KeyboardButton("Ya, rekomendasi lain")],
        [KeyboardButton("Tidak, selesai")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    await update.message.reply_text("Apakah Anda ingin rekomendasi lainnya?", reply_markup=reply_markup)

async def continue_recommendation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menangani pilihan pengguna untuk melanjutkan rekomendasi atau selesai."""
    user_choice = update.message.text

    if user_choice == "Ya, rekomendasi lain":
        keyboard = [[KeyboardButton(genre)] for genre in genres]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
        await update.message.reply_text("Pilih genre favorit Anda:", reply_markup=reply_markup)
    
    elif user_choice == "Tidak, selesai":
        await update.message.reply_text("Terima kasih telah menggunakan bot ini. Sampai jumpa!")

def main():
    print("Bot siap digunakan.")
    
    # Inisialisasi aplikasi bot Telegram
    app = ApplicationBuilder().token(os.environ.get('TELEGRAM_BOT_TOKEN')).build()
    
    # Tambahkan handler
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r'^(' + '|'.join(genres) + ')$'), genre_selected))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r'^\d{4}$'), year_input))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r'^(Ya, rekomendasi lain|Tidak, selesai)$'), continue_recommendation))

    print("Bot sedang berjalan...")
    # Jalankan bot dengan polling
    app.run_polling()

if __name__ == '__main__':
    main()
