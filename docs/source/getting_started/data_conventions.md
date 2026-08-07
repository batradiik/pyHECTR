# Data conventions

Unless stated otherwise, pyHECTR uses the following conventions:

- detector stacks have shape `(n_images, n_gamma, n_delta)`;
- mask stacks have the same shape as the corresponding detector stack;
- rod coordinates are ordered as
  `(image_index, gamma_pixel, delta_pixel)`;
- detector and sample rotation angles are expressed in degrees;
- pixel size and sample–detector distance must use consistent units;
- wavelengths are normally expressed in ångströms;
- HKL values are fractional reciprocal-lattice coordinates and are not
  automatically rounded to integer Miller indices.
