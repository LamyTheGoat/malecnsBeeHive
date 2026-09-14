#!/usr/bin/env Rscript
# Export the mushroom body wiring from the male CNS connectome.
#
# The Python model in ../flybrain runs on published population statistics by
# default.  This script replaces them with numbers measured in the actual
# reconstruction, using the malecns package from this repository.
#
#   Rscript beehive/scripts/export_mb_connectome.R [output_dir]
#
# Requires a neuprint token in NEUPRINT_TOKEN / the neuprint_token R
# environment variable -- see the README of this repository.  Then:
#
#   python3 -m flybrain info --connectome beehive/data/connectome
#
# Writes two CSVs that flybrain/connectome.py knows how to read:
#   mb_neurons.csv   bodyid, type, mbclass, claws
#   mb_kc_mbon.csv   bodyid_pre, bodyid_post, type_post, weight

suppressPackageStartupMessages({
  library(malecns)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[[1]] else "beehive/data/connectome"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

message("neuprint baglantisi kuruluyor ...")
conn <- mcns_neuprint()

fetch_meta <- function(pattern, label) {
  message(sprintf("  %s araniyor (%s) ...", label, pattern))
  meta <- mcns_neuprint_meta(pattern, conn = conn)
  message(sprintf("    %d noron", nrow(meta)))
  meta
}

kc   <- fetch_meta("/type:KC.*", "Kenyon hucreleri")
mbon <- fetch_meta("/type:MBON.*", "MBON'lar")
dan  <- fetch_meta("/type:(PAM|PPL1|PPL2).*", "dopaminerjik noronlar")

neurons <- bind_rows(
  kc   %>% transmute(bodyid, type, mbclass = "KC"),
  mbon %>% transmute(bodyid, type, mbclass = "MBON"),
  dan  %>% transmute(bodyid, type, mbclass = "DAN")
)

# Kenyon cell claws: distinct strong inputs in the calyx.  Anatomically a claw
# is a dendritic specialisation around one projection neuron bouton, so the
# count of distinct upstream partners is a reasonable proxy.
claws <- tryCatch({
  message("  KC girdileri sayiliyor (bu birkac dakika surebilir) ...")
  kc_in <- mcns_connection_table(kc$bodyid, partners = "inputs",
                                 threshold = 3L, conn = conn)
  kc_in %>%
    filter(!grepl("^KC", type) | is.na(type)) %>%
    count(bodyid, name = "claws")
}, error = function(e) {
  message("    KC girdileri alinamadi, atlaniyor: ", conditionMessage(e))
  tibble(bodyid = numeric(0), claws = integer(0))
})

neurons <- neurons %>% left_join(claws, by = "bodyid")

# KC -> MBON edges.  Querying from the MBON side is far cheaper: there are
# tens of MBONs and thousands of KCs.
message("  KC->MBON kenarlari cekiliyor ...")
kc_ids <- as.character(kc$bodyid)
edges <- mcns_connection_table(mbon$bodyid, partners = "inputs",
                               threshold = 1L, conn = conn) %>%
  filter(as.character(partner) %in% kc_ids) %>%
  left_join(mbon %>% transmute(bodyid, type_post = type), by = "bodyid") %>%
  transmute(bodyid_pre = partner, bodyid_post = bodyid, type_post, weight)

message(sprintf("    %d KC->MBON kenari, %d MBON tipi",
                nrow(edges), dplyr::n_distinct(edges$type_post)))

write.csv(neurons, file.path(outdir, "mb_neurons.csv"), row.names = FALSE)
write.csv(edges, file.path(outdir, "mb_kc_mbon.csv"), row.names = FALSE)

message("Yazildi: ", normalizePath(outdir))
message("Simdi: python3 -m flybrain info --connectome ", outdir)
