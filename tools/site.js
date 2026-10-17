/* Navigation helpers shared by every page: sidebar filter, "/" to search, and
   on the index page the search box, tag filter, sorting and the URL hash that
   remembers them. No dependencies; the pages stay readable with this disabled. */
(function () {
  "use strict";
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  // ---- sidebar filter (every page)
  var nav = $("#site-nav"), nf = $(".nav-filter");
  if (nav && nf) {
    nf.addEventListener("input", function () {
      var terms = nf.value.toLowerCase().split(/\s+/).filter(Boolean), cat = null, sub = null, list = [];
      var kids = $$(".nav-cat, .nav-sub, ul", nav), groups = [];
      kids.forEach(function (el) {
        if (el.classList.contains("nav-cat")) { cat = { el: el, subs: [], any: false }; groups.push(cat); sub = null; }
        else if (el.classList.contains("nav-sub")) { sub = { el: el, any: false }; cat.subs.push(sub); }
        else {
          var any = false;
          $$("li", el).forEach(function (li) {
            var t = li.textContent.toLowerCase();
            var ok = terms.every(function (w) { return t.indexOf(w) >= 0; });
            li.hidden = !ok; any = any || ok;
          });
          el.hidden = !any; if (sub) { sub.any = sub.any || any; } cat.any = cat.any || any;
        }
      });
      groups.forEach(function (g) { g.el.hidden = !g.any; g.subs.forEach(function (s) { s.el.hidden = !s.any; }); });
    });
  }

  // "/" focuses whichever search box the page has
  document.addEventListener("keydown", function (e) {
    if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target.tagName; if (t === "INPUT" || t === "TEXTAREA" || t === "SELECT") return;
    var box = $("#q") || nf; if (!box) return;
    if (document.documentElement.classList.contains("nav-collapsed") && !$("#q")) {
      var b = $(".nav-toggle"); if (b) b.click();
    }
    e.preventDefault(); box.focus(); box.select();
  });

  // ---- tag buttons on a paper page jump to the index filtered by that tag
  var results = $("#results");
  if (!results) {
    $$(".paper-head .tag").forEach(function (b) {
      b.addEventListener("click", function () {
        var home = $(".nav-home"); if (!home) return;
        location.href = home.getAttribute("href") + "#tag=" + encodeURIComponent(b.dataset.tag);
      });
    });
    return;
  }

  // ---- index page
  var q = $("#q"), kind = $("#f-kind"), status = $("#f-status"), sort = $("#f-sort");
  var count = $("#count"), reset = $("#reset"), none = $("#none");
  var items = $$("li[data-kind]", results), tags = [];
  var chips = $$(".chip");

  function readHash() {
    var h = {}; location.hash.replace(/^#/, "").split("&").forEach(function (kv) {
      var i = kv.indexOf("="); if (i > 0) h[kv.slice(0, i)] = decodeURIComponent(kv.slice(i + 1));
    });
    q.value = h.q || ""; kind.value = h.kind || ""; status.value = h.status || ""; sort.value = h.sort || "new";
    tags = h.tag ? h.tag.split(",").filter(Boolean) : [];
  }
  function writeHash() {
    var p = [];
    if (q.value) p.push("q=" + encodeURIComponent(q.value));
    if (tags.length) p.push("tag=" + encodeURIComponent(tags.join(",")));
    if (kind.value) p.push("kind=" + kind.value);
    if (status.value) p.push("status=" + status.value);
    if (sort.value !== "new") p.push("sort=" + sort.value);
    history.replaceState(null, "", p.length ? "#" + p.join("&") : location.pathname + location.search);
  }
  function key(li, mode) {
    var y = parseInt(li.dataset.year, 10); if (isNaN(y)) y = mode === "old" ? 9999 : -9999;
    return y;
  }
  function apply() {
    var terms = q.value.toLowerCase().split(/\s+/).filter(Boolean), shown = 0;
    items.forEach(function (li) {
      var own = (li.dataset.tags || "").split("|");
      var ok = terms.every(function (w) { return li.dataset.text.indexOf(w) >= 0; }) &&
        tags.every(function (t) { return own.indexOf(t) >= 0; }) &&
        (!kind.value || li.dataset.kind === kind.value) &&
        (!status.value || li.dataset.status === status.value);
      li.hidden = !ok; if (ok) shown++;
    });
    $$("ul.papers", results).forEach(function (ul) {
      var lis = $$("li", ul);
      lis.sort(function (a, b) {
        if (sort.value === "title") return a.dataset.title < b.dataset.title ? -1 : 1;
        var d = key(b, "new") - key(a, "new"); if (sort.value === "old") d = -d;
        return d || (a.dataset.title < b.dataset.title ? -1 : 1);
      }).forEach(function (li) { ul.appendChild(li); });
      ul.parentNode.hidden = !lis.some(function (li) { return !li.hidden; });
    });
    $$("section.cat", results).forEach(function (s) {
      s.hidden = !$$(".sub", s).some(function (d) { return !d.hidden; });
    });
    chips.forEach(function (c) { c.classList.toggle("on", tags.indexOf(c.dataset.tag) >= 0); });
    $$(".tag", results).forEach(function (c) { c.classList.toggle("on", tags.indexOf(c.dataset.tag) >= 0); });
    var active = terms.length || tags.length || kind.value || status.value;
    count.textContent = active ? shown + " of " + items.length + " pages" : items.length + " pages";
    reset.hidden = !active; none.hidden = shown > 0;
    if (tags.length) { var d = $(".chips"); if (d) d.open = true; }
    writeHash();
  }
  function toggleTag(t) {
    var i = tags.indexOf(t); if (i >= 0) tags.splice(i, 1); else tags.push(t); apply();
  }
  [q, kind, status, sort].forEach(function (el) { el.addEventListener("input", apply); });
  chips.forEach(function (c) { c.addEventListener("click", function () { toggleTag(c.dataset.tag); }); });
  results.addEventListener("click", function (e) {
    var t = e.target.closest ? e.target.closest(".tag") : null;
    if (t) { toggleTag(t.dataset.tag); return; }
    var s = e.target.closest ? e.target.closest(".summary") : null;
    if (s) s.classList.toggle("open");
  });
  results.addEventListener("keydown", function (e) {
    if ((e.key === "Enter" || e.key === " ") && e.target.classList.contains("summary")) {
      e.preventDefault(); e.target.classList.toggle("open");
    }
  });
  reset.addEventListener("click", function () { q.value = ""; kind.value = ""; status.value = ""; tags = []; apply(); });
  window.addEventListener("hashchange", function () { readHash(); apply(); });
  readHash(); apply();
})();
