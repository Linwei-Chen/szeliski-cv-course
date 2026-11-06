/* deck.js —— 幻灯片状态驱动
 * URL 参数：?slide=<index>&step=<k>&sub=<urlencoded 字幕文本>
 * 行为：仅显示第 slide 个 <section class="slide">；其内 .step 元素按 data-step<=k 显现；
 *       sub 非空时显示底部字幕条。供无头浏览器逐帧截图使用，也支持浏览器放映（无参数时显示第一页）。
 */
(function () {
  const q = new URLSearchParams(location.search);
  const slideIdx = parseInt(q.get("slide") || "0", 10);
  const step = parseInt(q.get("step") || "99", 10);
  const sub = q.get("sub");
  const slides = document.querySelectorAll(".slide");
  slides.forEach((s, i) => s.classList.toggle("active", i === slideIdx));
  const cur = slides[slideIdx];
  if (cur) {
    cur.querySelectorAll(".step").forEach((el) => {
      const k = parseInt(el.dataset.step || "1", 10);
      el.classList.toggle("on", k <= step);
    });
  }
  const bar = document.getElementById("subbar");
  if (bar) {
    if (sub) {
      document.body.classList.add("hassub");
      bar.classList.add("on");
      // 术语双语高亮：英文词转粗体强调色
      bar.innerHTML = "<span>" +
        sub.replace(/[A-Za-z][A-Za-z0-9\-']*/g, (m) => "<b>" + m + "</b>") +
        "</span>";
    }
  }
  // 顶部进度条：帧进度 =（此前各页步数之和 + 当前步）/ 总步数
  const total = [...slides].reduce((n, s) => n + (parseInt(s.dataset.steps || "1", 10) || 1), 0);
  const before = [...slides].slice(0, slideIdx)
    .reduce((n, s) => n + (parseInt(s.dataset.steps || "1", 10) || 1), 0);
  const done = Math.min(1, Math.max(0, (before + Math.min(step, 999)) / total));
  const prog = document.createElement("div");
  prog.id = "progress";
  prog.style.width = (done * 100).toFixed(2) + "%";
  document.body.appendChild(prog);
})();
