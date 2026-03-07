# =============================================================================
# Assets directory for STT benchmarking
# =============================================================================
#
# This directory stores audio datasets used for speech-to-text evaluation.
#
# Setup:
#   bash assets/download_librispeech.sh
#
# Contents (after download):
#   librispeech-test-clean/     LibriSpeech test-clean subset
#     LibriSpeech/test-clean/
#       <speaker-id>/
#         <chapter-id>/
#           *.flac               Audio files (16kHz, mono)
#           *.trans.txt          Reference transcriptions
#
# The librispeech-test-clean/ directory is .gitignored.
