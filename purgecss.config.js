module.exports = {
  content: ["_site/**/*.html", "_site/**/*.js"],
  css: ["_site/assets/css/*.css"],
  output: "_site/assets/css/",
  skippedContentGlobs: ["_site/assets/**/*.html"],
  safelist: [
    // Font-related
    /font/,
    /^ws-/,
    // Bootstrap utilities
    /^(text|bg|border|display|flex|grid|position|float)-/,
    // Common utility prefixes
    /^m[trblxy]?-/,
    /^p[trblxy]?-/,
    /^w-/,
    /^h-/,
  ],
};
