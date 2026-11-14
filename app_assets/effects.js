// ===================== Hieu ung Giang Sinh =====================
(function () {
  if (window.__xmasFx) return;
  window.__xmasFx = true;

  function start() {
    var body = document.body;
    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // ---------- lop nen: tuyet + trang tri ----------
    var fx = document.createElement("div");
    fx.id = "xmas-fx";
    var back = document.createElement("canvas");
    fx.appendChild(back);
    body.appendChild(fx);

    var front = document.createElement("canvas");
    front.id = "xmas-front";
    body.appendChild(front);

    // Trang tri hai goc duoi
    function deco(html, css) {
      var d = document.createElement("div");
      d.className = "deco";
      d.innerHTML = html;
      d.style.cssText = css;
      fx.appendChild(d);
    }
    deco('<span class="sway">🎄</span>', "left:1%;font-size:120px;bottom:-6px");
    deco('<span class="sway" style="animation-delay:-1.3s">⛄</span>', "left:9%;font-size:78px");
    deco("🎁", "left:15%;font-size:46px;bottom:4px");
    deco('<span class="hop">🦌</span>', "right:13%;font-size:70px;bottom:6px");
    deco("🎁", "right:9%;font-size:52px;bottom:2px");
    deco('<span class="sway" style="animation-delay:-0.7s">🎄</span>', "right:1%;font-size:130px;bottom:-6px");
    deco("🕯️", "left:21%;font-size:34px;bottom:8px");
    deco("🍬", "right:20%;font-size:34px;bottom:8px");

    // Sao lap lanh
    var twk = ["⭐", "✨", "🌟", "❄️", "✨"];
    for (var i = 0; i < 16; i++) {
      var s = document.createElement("div");
      s.className = "twinkle";
      s.textContent = twk[i % twk.length];
      s.style.left = Math.random() * 96 + "%";
      s.style.top = 6 + Math.random() * 55 + "%";
      s.style.fontSize = 12 + Math.random() * 16 + "px";
      s.style.animationDelay = -Math.random() * 2.2 + "s";
      fx.appendChild(s);
    }

    // ---------- xe tuan loc bay ngang ----------
    var sleigh = document.createElement("div");
    sleigh.id = "sleigh";
    sleigh.innerHTML =
      '<span class="bob">🦌</span><span class="bob">🦌</span><span class="bob">🦌</span>' +
      '<span class="santa">🛷<span>🎅</span></span>';
    var sky = document.createElement("div"); // lop tren cung: xe bay ngang qua tieu de
    sky.id = "sleigh-layer";
    sky.appendChild(sleigh);
    body.appendChild(sky);
    var sx = window.innerWidth * 0.62, sy = 0, lastSpark = 0;

    // ---------- day den ----------
    var lights = document.createElement("div");
    lights.id = "xmas-lights";
    body.appendChild(lights);
    function buildLights() {
      var w = window.innerWidth, seg = 160, gap = 34;
      var colors = ["#ff2a3d", "#2bd45a", "#ffd23f", "#3fa9ff", "#ff7af0"];
      var pts = [], html = "";
      function yAt(x) { return 6 + 16 * Math.sin(Math.PI * ((x % seg) / seg)); }
      for (var x = 0; x <= w; x += 8) pts.push(x + "," + yAt(x).toFixed(1));
      html += '<svg width="' + w + '" height="48"><polyline points="' + pts.join(" ") +
        '" fill="none" stroke="#1d3b1d" stroke-width="2.5"/></svg>';
      var k = 0;
      for (var bx = gap / 2; bx < w; bx += gap, k++) {
        var c = colors[k % colors.length];
        html += '<div class="bulb" style="left:' + bx + "px;top:" + (yAt(bx) + 3).toFixed(1) +
          "px;background:" + c + ";color:" + c + ";animation-delay:" + (-(k % 4) * 0.35) + 's"></div>';
      }
      lights.innerHTML = html;
    }
    buildLights();

    // ---------- tuyet ----------
    var W, H, ctxB = back.getContext("2d"), ctxF = front.getContext("2d");
    var flakesB = [], flakesF = [];
    function resize() {
      W = window.innerWidth; H = window.innerHeight;
      [back, front].forEach(function (c) { c.width = W; c.height = H; });
      buildLights();
    }
    function mk(big) {
      return {
        x: Math.random() * W, y: Math.random() * H,
        r: big ? 2.5 + Math.random() * 3 : 0.8 + Math.random() * 2.2,
        vy: big ? 0.9 + Math.random() * 1.2 : 0.4 + Math.random() * 1,
        ph: Math.random() * 6.28, amp: 0.3 + Math.random() * 0.8,
        emoji: big && Math.random() < 0.35
      };
    }
    resize();
    for (var a = 0; a < 170; a++) flakesB.push(mk(false));
    for (var b = 0; b < 22; b++) flakesF.push(mk(true));
    window.addEventListener("resize", resize);

    function drawFlakes(ctx, arr, t) {
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      for (var i = 0; i < arr.length; i++) {
        var f = arr[i];
        f.y += f.vy; f.x += Math.sin(t / 900 + f.ph) * f.amp;
        if (f.y > H + 10) { f.y = -10; f.x = Math.random() * W; }
        if (f.emoji) {
          ctx.font = f.r * 6 + "px serif";
          ctx.fillText("❄", f.x, f.y);
        } else {
          ctx.beginPath(); ctx.arc(f.x, f.y, f.r, 0, 6.283); ctx.fill();
        }
      }
    }

    var off = false;
    try { off = localStorage.getItem("xmas-fx-off") === "1"; } catch (e) {}
    if (off || reduce) body.classList.add("fx-off");

    var lastT = 0;
    function loop(t) {
      var dt = Math.min(t - lastT, 50); lastT = t;
      if (!body.classList.contains("fx-off") && !body.classList.contains("xmas-busy") && !document.hidden) {
        drawFlakes(ctxB, flakesB, t);
        drawFlakes(ctxF, flakesF, t);
        // xe tuan loc: bay tu phai sang trai, nhap nho (toc do theo thoi gian, khong phu thuoc tan so man hinh)
        sx -= dt * 0.16;
        if (sx < -sleigh.offsetWidth - 80) { sx = W + 120 + Math.random() * 400; }
        sy = Math.sin(t / 700) * 18;
        sleigh.style.transform = "translate(" + sx + "px," + sy + "px)";
        if (t - lastSpark > 110 && sx < W && sx > -sleigh.offsetWidth) {
          lastSpark = t;
          var sp = document.createElement("div");
          sp.className = "spark";
          sp.textContent = Math.random() < 0.5 ? "✨" : "⭐";
          sp.style.left = sx + sleigh.offsetWidth - 10 + "px";
          sp.style.top = sleigh.offsetTop + sy + 20 + Math.random() * 20 + "px";
          sky.appendChild(sp);
          setTimeout(function (n) { return function () { n.remove(); }; }(sp), 1600);
        }
      }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);

    // ---------- nut bat/tat ----------
    var tg = document.createElement("button");
    tg.id = "fx-toggle";
    tg.title = "Bật/tắt hiệu ứng";
    tg.textContent = "❄️";
    tg.onclick = function () {
      body.classList.toggle("fx-off");
      var o = body.classList.contains("fx-off");
      try { localStorage.setItem("xmas-fx-off", o ? "1" : "0"); } catch (e) {}
      if (o) { ctxB.clearRect(0, 0, W, H); ctxF.clearRect(0, 0, W, H); }
    };
    body.appendChild(tg);

    // ---------- keo tha video: chi lay DUONG DAN (Python xu ly su kien drop) ----------
    window.xmasSetPath = function (p) {
      var el = document.querySelector("#xmas-path textarea, #xmas-path input");
      if (!el) return;
      var setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), "value").set;
      setter.call(el, p);
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
      el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      el.blur();
    };
    var ov = document.createElement("div");
    ov.id = "drop-overlay";
    ov.innerHTML = "<div>🎁 Thả video vào đây<br><small>App chỉ đọc đường dẫn – video dài bao nhiêu cũng được</small></div>";
    body.appendChild(ov);
    var dragDepth = 0;
    function hasFiles(e) { return e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types || [], "Files") >= 0; }
    document.addEventListener("dragenter", function (e) { if (hasFiles(e)) { dragDepth++; body.classList.add("dragging"); } });
    document.addEventListener("dragleave", function () { if (--dragDepth <= 0) { dragDepth = 0; body.classList.remove("dragging"); } });
    document.addEventListener("dragover", function (e) { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } });
    document.addEventListener("drop", function (e) {
      e.preventDefault(); // khong de cua so mo/tai file
      dragDepth = 0; body.classList.remove("dragging");
      if (!window.pywebview) alert("Đang chạy bằng trình duyệt: hãy dán đường dẫn file vào ô 'Video nguồn'.");
    });

    // ---------- mua qua khi xu ly xong ----------
    window.xmasCelebrate = function () {
      var items = ["🎁", "🎉", "⭐", "🎄", "🎅", "❄️", "🍬", "🔔"];
      for (var i = 0; i < 60; i++) {
        var c = document.createElement("div");
        c.className = "confetti";
        c.textContent = items[i % items.length];
        c.style.left = Math.random() * 100 + "vw";
        c.style.animationDuration = 2.5 + Math.random() * 2.5 + "s";
        c.style.animationDelay = Math.random() * 1.2 + "s";
        document.body.appendChild(c);
        setTimeout(function (n) { return function () { n.remove(); }; }(c), 6500);
      }
    };
    var seen = false;
    new MutationObserver(function () {
      var st = document.getElementById("xmas-status");
      var txt = st ? st.textContent : "";
      var done = txt.indexOf("Xong!") >= 0;
      // dang xu ly -> tam dung hieu ung de danh toan bo GPU cho AI (hieu ung an ~20% GPU)
      body.classList.toggle("xmas-busy", txt.indexOf("⏳") >= 0 || txt.indexOf("Đang tối ưu") >= 0);
      if (done && !seen) window.xmasCelebrate();
      seen = done;
    }).observe(body, { childList: true, subtree: true, characterData: true });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
