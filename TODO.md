TODO List:
- [ ] text output
- [x] picture input
- [x] svg output
- [ ] GUI to sample output with sliders for
    - [ ] gaussian block size and constant
    - [ ] edge min/maxval
- [ ] make `hatch` SVG output efficient (compute boundaries rather than relying on a clipping mask, i.e. preclip)
- [ ] Gaussian Paint Strokes
- [ ] GDAL to import topography
- [x] Make into a module

Possible Archs:
- Command Line (Hard to use)
- Native App (Non-Portable, probably)
- JS Rewrite (Slow)
- Server/Client App Mix (Complex & Costly, but no more complex than native)

# Goal
Rewrite into C++, then add interaction with Express.js based server.

Also could just use Python web framework to make it easier. But that's lame and slower, but I could probably also do both (ensuring the templates are compatible).
