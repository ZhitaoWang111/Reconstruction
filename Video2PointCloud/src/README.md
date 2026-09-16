# Local integration code

- `common/`: shared ScanNet evaluation and point-cloud utilities.
- `amb3r/`: AMB3R frontend/VO wrappers, scale post-processing, extraction, and evaluation.
- `da3/`: Depth Anything 3 Nested runners and DA3-Streaming conversion.
- `mapanything/`: MapAnything single-pass and streaming wrappers.
- `single_frame_depth/`: ten-frame AMB3R/DA3 depth comparison.
- `visualization/`: source and builder for the interactive point-cloud page.

These files call the upstream repositories in `../ThirdParty/`; no local integration code belongs inside an upstream checkout.
