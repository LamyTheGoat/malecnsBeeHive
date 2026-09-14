# BeeHive — begenilerinizi ogrenen sanal bir meyve sinegi beyni

Bir *Drosophila* beyninin ogrenme devresini kurar, ona fotograflar gosterir ve
sag/sol kaydirmalarinizla egitirsiniz. Devre uydurma degil: sinegin gercek
cagrisimsal bellek devresi (mantar cismi) ne yapiyorsa kod da onu yapiyor.

```
foto -> 756 ommatidyum -> lamina/medulla -> 2000 Kenyon hucresi
     -> 2 MBON (yaklas / kac) -> DNa02 sol/sag -> kaydirma
                      ^
                      |
             dopamin (sizin kaydirmaniz)
```

## 30 saniyede dene

```bash
cd beehive
pip install -r requirements.txt
python3 -m flybrain demo
```

Sentetik profillerde gizli bir "zevk kurali" vardir, sinek onu bilmez ve
kaydirmalardan ogrenir:

```
Test dogrulugu (17 olcum, egitim boyunca):
  ▂▁█▇█████▆▇▇▇█▇▇█
  54 52 76 73 76 77 77 76 75 70 74 74 74 77 71 73 75
  naif sinek %54  ->  egitilmis sinek %75
```

## Kendi fotograflariniz ile

```bash
python3 -m flybrain eye   --image foto.jpg              # sinek ne goruyor?
python3 -m flybrain train --photos ./fotolar --brain fly.npz
python3 -m flybrain swipe --photos ./yeniler --brain fly.npz
```

`train` her fotografi ASCII olarak gosterir, kendi tahminini soyler, sonra size
sorar: `j` = sol, `l` = sag, `s` = atla, `q` = cik. Kaydirmalariniz fotograf
klasorunde `swipes.json` olarak birikir; `--batch` ile soru sormadan sadece
etiketlenmislerle yeniden egitebilirsiniz.

Egitilen beyin `fly.npz` icinde durur, uzerine egitmeye devam edebilirsiniz.

## Devrede ne var

| Biyoloji | Kod | Gercek sayi |
|---|---|---|
| Ommatidyum, altigen orgu | `eye.py`, `build_lattice` | ~750-800 / goz |
| R1-6 genis bantli fotoreseptor | `look()["lum"]`, log sikistirma | — |
| R7/R8 pale / yellow | `pale`, `yellow`, `opponent` | — |
| Lamina L1/L2, ON/OFF Weber kontrasti | `on`, `off` | — |
| Medulla, 3 ana altigen eksen | `axis_h/p/q` | — |
| Lobula VPN'leri, retinotopik havuzlama | `vpn()` | 144 kanal (bkz. "kaytaklar") |
| Kenyon hucreleri, rastgele 6 pence | `MushroomBody.kc_inputs` | ~2000 KC, 6-7 pence |
| APL geri besleme inhibisyonu | `kenyon_cells()` esikleme | KC'lerin ~%5-10'u aktif |
| KC->MBON sinapsi | `w_out` | — |
| Dopamin kapili **depresyon** | `teach()` | — |
| MBON tahterevallisi (yaklas vs kac) | `valence()` | ~35 MBON tipinin 2'si |
| DNa02, sol/sag donus | `steering.py` | — |

Ogrenme kuralinin yonu onemli: sinek Hebb tarzi guclendirmeyle degil,
**depresyonla** ogrenir. Naif sinekte tum KC->MBON sinapslari esit ve gucludur;
dopamin o an aktif olan Kenyon hucrelerinin sinapsini *zayiflatir*. Yani sinek
"neyi sevecegini" ogrenmez, baslangicta her seye acikken **neyi elemesi
gerektigini** oyar. Sag kaydirma odul DAN'larini (PAM) atesler ve *kacinma*
MBON'unu zayiflatir; sol kaydirma ceza DAN'larini (PPL1) atesler ve *yaklasma*
MBON'unu zayiflatir. Geriye kalan dengesizlik donus komutudur.

### Kaytaklar (dogrulugu adina)

- Gercek sinekte gorsel bilgi mantar cismine cok dar bir yoldan, on kadar
  VPN-MB / ME-MB noronu uzerinden ve sadece γd Kenyon hucrelerine girer
  (Vogt ve ark. 2016). Burada 144 kanal kullaniyorum — bilerek genis tuttum,
  yoksa ogrenecek bir sey kalmiyor.
- ~35 MBON tipi yerine 2 kanal, tek yarikure, zaman yok: fotograf hareket
  etmedigi icin T4/T5 hareket ve LPLC2 carpma kanallari yok.
- Ates dizisi (spike) yok; hepsi hiz kodu.

## Sinek gercekte neyi ogrenebilir

Bu projenin en ilginc tarafi su: **sinegin optigi tavani belirliyor.**
Ommatidyumlar arasi aci ~5 derece, yani insan foveasinin ~1/100'u. Bir sinek
telefon ekranindaki yuze bakmaz; kaba renk, parlaklik ve kontrast lekeleri
gorur. Demo bunu her calistirmada olcup basar:

```
  sans seviyesi                       : %50
  sineğin optigi neye izin veriyor    : %72
  etiket gurultusu tavani             : %90
```

Ortadaki satir, gizli zevk kuralinin ne kadarinin ommatidyum orgusunden sag
cikabildigidir. Sinek %75'e ulasiyor, yani **ogrenme kurali gorebildigi her
seyi sikip cikariyor**; kalan bosluk optigin kaybi. Kendi fotograflarinizda da
beklenti bu olmali: sinek "tip" degil, fotograflarinizin kaba istatistiklerini
(sicak/soguk ton, aydinlik, kontrast, kalabalik kompozisyon) ogrenir. Zevkiniz
bunlarla ne kadar orortusuyorsa o kadar isabet eder.

Bu bir kusur degil, sonucun kendisi: sinek beyni bir yuz siniflandiricisi
degil, cok az noronla cok hizli karar veren bir valans makinesidir.

## Gercek konnektom verisini bagla

Varsayilan sayilar yayinlardan gelir. Bu deponun sardigi male CNS
rekonstruksiyonundan gercek sayilari cekmek icin:

```bash
Rscript beehive/scripts/export_mb_connectome.R beehive/data/connectome
python3 -m flybrain info --connectome beehive/data/connectome
```

Betik `malecns` paketinin kendi fonksiyonlarini kullanir
(`mcns_neuprint_meta`, `mcns_connection_table`), Kenyon hucrelerini,
MBON'lari ve dopaminerjik noronlari bulur, KC->MBON kenarlarini iki CSV'ye
yazar. Neuprint token'i gerekir — kurulumu deponun ana `README.md`'sinde.

> Bu betik canli bir neuprint sunucusuna karsi calistirilmadi: gelistirme
> ortaminda R kurulu degildi ve token yoktu. Mantigi ve paket fonksiyon
> imzalari dogru, ama ilk calistirmada tip adi regex'lerini (`/type:KC.*`)
> veri setindeki gercek adlarla karsilastirin.

## Bumble hakkinda

Uygulamayi otomatik kaydirmayin. Bumble'in kullanim sartlari otomasyonu
yasaklar ve bot tespiti hesabi kapatir. Bu proje yerel bir fotograf klasoru
uzerinde calisir; kararlar terminalde cikar, uygulamaya dokunmaz. Egitilmis
sinegi "ikinci bir goz" olarak kullanin, eliniz sizde kalsin.

## Testler

```bash
cd beehive
python3 -m unittest discover -s tests
```

29 test; ogrenme kuralinin yonunu, seyrek kodlamayi, agirliklarin asla
guclenmemesini, pozlamadan bagimsizligi ve egitilmis sinegin naif sinegi
gectigini kontrol eder.

## Alana nasil girilir

**Veri.** [neuprint.janelia.org](https://neuprint.janelia.org) (hemibrain,
male CNS), [FlyWire](https://flywire.ai) (tam disi beyni),
[Virtual Fly Brain](https://virtualflybrain.org). Hepsi tarayicidan,
kayitsiz gezilebilir.

**Arac.** R tarafinda bu depo + `neuprintr` + `natverse`; Python tarafinda
`neuprint-python`, `navis`, `fafbseg`.

**Simulasyon.** "Konnektomdan davranisa" isini yapan calismalar:
Lappalainen ve ark. 2024 (konnektom-kisitli gorme agi), NeuroMechFly
(Lobato-Rios 2022, Wang-Chen 2024 — bedenlenmis sinek simulasyonu). Gordugunuz
"araba park eden sinek" demolari bu ailedendir.

**Mantar cismi okuma listesi.** Aso ve ark. 2014 (bolme haritasi), Caron ve
ark. 2013 (rastgele pence baglantisi), Lin ve ark. 2014 (APL seyrekligi),
Hige ve ark. 2015 ve Cohn ve ark. 2015 (depresyon ve dopamin kapisi),
Handler ve ark. 2019 (zamanlamaya bagli isaret), Modi, Shuai ve Turner 2020
(derleme — buradan baslayin).

Kaynaklari yaziya dokmeden once kendiniz dogrulayin; yil ve dergi
hatirlamalarim yanilabilir.

## "Sinekler neden bazi insanlara daha cok konuyor?"

Dikkat: bu soruyu iyi calisilmis haliyle **sivrisinekler** icin sorabilirsiniz
— orada kisiler arasi fark gercek ve olculmustur (vucut kokusu profili, cilt
mikrobiyotasi, CO2 verimi, kan grubu tartismali). *Drosophila* icin kisiden
kisiye farki gosteren denetimli bir literatur pratikte yok. Bilinenler tur
duzeyinde:

- **Fermentasyon kokulari.** Meyve sinegini ceken sey esas olarak maya ve
  fermentasyon ucucularidir: asetik asit, etanol, esterler. Uzerinizde bira,
  sarap, meyve suyu, sirke ya da olgun meyve kalintisi varsa hedefsiniz.
- **Ter ve cilt mikrobiyotasi.** Terin kendisi degil, ciltteki bakterilerin
  teri islemesiyle cikan ucucular. Bu, kisiden kisiye farkin en olasi kaynagi.
- **Sicaklik ve nem.** Sinekler ~25-30 °C'yi tercih eder; sicak, nemli bir cilt
  inis icin uygun bir yuzeydir.
- **CO2.** Gorunduğu kadar basit degil: yuruyen *Drosophila* icin CO2 kacinma
  uyaranidir (Suh ve ark. 2004), ucusta baglam degistirir. Sivrisinekteki
  "nefesiniz sizi ele veriyor" hikayesi meyve sinegine dogrudan tasinmaz.
- **Kontrast.** Koyu, yuksek kontrastli, hareketsiz bir nesne inis hedefi
  olarak caziptir — bu projenin `on`/`off` kanallarinin gordugu seyin aynisi.

Yani muhtemelen sizi secen sey kim oldugunuz degil, o gun uzerinizde ne
koktugu. Merak ettiginiz sey buysa gercek deney kurulabilir: iki kisi, ayni
odada, elde tutulan birer sirke tuzagi, sayim. Sinegin cevabini bu projedeki
devreye sormayin — o sadece goruyor, koklamiyor.
