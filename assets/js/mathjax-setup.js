window.MathJax = {
  tex: {
    tags: "ams",
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
  },
  startup: {
    pageReady: function () {
      return MathJax.startup.defaultPageReady().then(function () {
        fitDisplayMath();
        var timer;
        window.addEventListener("resize", function () {
          clearTimeout(timer);
          timer = setTimeout(fitDisplayMath, 150);
        });
      });
    },
  },
  options: {
    renderActions: {
      addCss: [
        200,
        function (doc) {
          const style = document.createElement("style");
          style.innerHTML = `
          .mjx-container {
            color: inherit;
          }
        `;
          document.head.appendChild(style);
        },
        "",
      ],
    },
  },
};

function fitDisplayMath() {
  var nodes = document.querySelectorAll('mjx-container[display="true"]');
  for (var i = 0; i < nodes.length; i++) {
    var el = nodes[i];
    if (!el.dataset.baseFontSize) {
      el.dataset.baseFontSize = window.getComputedStyle(el).fontSize;
    }
    el.style.fontSize = el.dataset.baseFontSize;
    var available = el.clientWidth;
    var needed = el.scrollWidth;
    if (available > 0 && needed > available + 1) {
      var base = parseFloat(el.dataset.baseFontSize);
      var scale = Math.max(available / needed, 0.5);
      el.style.fontSize = base * scale * 0.98 + "px";
    }
  }
}
