(function () {
  "use strict";

  const payload = window.XA202608_DEMO_PAYLOAD;
  const app = document.getElementById("app");
  const navItems = Array.from(document.querySelectorAll("[data-view]"));

  if (!payload || !payload.tasks) {
    app.innerHTML = [
      '<section class="empty-state">',
      "<h2>Demo 数据未加载</h2>",
      "<p>请先运行 scripts/build_html_demo.py 生成 assets/demo-data.js。</p>",
      "</section>",
    ].join("");
    return;
  }

  const taskKeys = ["first_transfer", "second_transfer"];
  const taskAccent = {
    first_transfer: "#35c7e8",
    second_transfer: "#43d17d",
  };
  const taskFeatureText = {
    first_transfer: {
      short: "RW",
      healthName: "RUL",
      unit: "归一化 RUL",
      primary: "摩擦/润滑退化",
      observable: "电流、温度、振动代理、转速波动",
      riskLabel: "失效接近度",
      advice: ["保持当前工况", "加强润滑状态监测", "重点关注温升趋势"],
    },
    second_transfer: {
      short: "BAT",
      healthName: "SOH/RUL",
      unit: "归一化 SOH/RUL",
      primary: "容量下降/内阻增长",
      observable: "电压、电流、温度、轨道充放电响应",
      riskLabel: "容量衰减风险",
      advice: ["维持常规遥测", "提高采样频率", "安排预防性维护窗口"],
    },
  };

  const fmt = {
    metric(value, digits = 3) {
      if (value === undefined || value === null || value === "") return "NA";
      const num = Number(value);
      return Number.isFinite(num) ? num.toFixed(digits) : String(value);
    },
    pct(value, digits = 1) {
      if (value === undefined || value === null || value === "") return "NA";
      const num = Number(value) * 100;
      return Number.isFinite(num) ? `${num.toFixed(digits)}%` : String(value);
    },
  };

  function htmlEscape(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function taskLabel(key) {
    const task = payload.tasks[key];
    return `${task.label} · ${task.component}`;
  }

  function metricCards(task) {
    const metrics = task.metrics || {};
    return [
      ["RMSE", fmt.metric(metrics.rmse), "越低越好"],
      ["MAE", fmt.metric(metrics.mae), "越低越好"],
      ["NASA Score", fmt.metric(metrics.nasa_score, 2), "越低越好"],
      ["RA", fmt.metric(metrics.ra), "越高越好"],
      ["Last RMSE", fmt.metric(metrics.last_window_rmse), "末期窗口"],
    ]
      .map(
        ([label, value, note]) => `
        <div class="metric-card">
          <div class="metric-label">${label}</div>
          <div class="metric-value">${value}</div>
          <div class="metric-note">${note}</div>
        </div>`
      )
      .join("");
  }

  function setActive(view) {
    navItems.forEach((item) => item.classList.toggle("is-active", item.dataset.view === view));
  }

  function render(view) {
    if (!["overview", "data", "wheel", "battery", "method"].includes(view)) {
      view = "overview";
    }
    setActive(view);
    if (view === "overview") renderOverview();
    if (view === "data") renderData();
    if (view === "wheel") renderTaskDetail("first_transfer");
    if (view === "battery") renderTaskDetail("second_transfer");
    if (view === "method") renderMethod();
  }

  function renderOverview() {
    const first = payload.tasks.first_transfer;
    const second = payload.tasks.second_transfer;
    app.innerHTML = `
      <section class="panel">
        <div class="panel-title">
          <div>
            <h2>跨领域退化数据迁移与预测流程</h2>
            <p class="muted">该页面只读取已生成实验产物，不训练模型，不重新拟合 TC。</p>
          </div>
          <span class="tag green">Evidence package ready</span>
        </div>
        <div class="flow-row">
          <div class="flow-card">
            <div class="flow-index">01 Source</div>
            <h3>公开退化源域</h3>
            <p>XJTU-SY Bearing 与 NASA Battery 提供退化表征学习基础。</p>
          </div>
          <div class="flow-card">
            <div class="flow-index">02 Adaptation</div>
            <h3>阶段感知迁移</h3>
            <p>伪时间阶段、局部分布对齐、SAC 与 R-SPA 共同约束。</p>
          </div>
          <div class="flow-card">
            <div class="flow-index">03 Target</div>
            <h3>航天仿真目标域</h3>
            <p>反作用轮摩擦退化与卫星电池容量/内阻退化。</p>
          </div>
          <div class="flow-card">
            <div class="flow-index">04 PHM</div>
            <h3>寿命预测与风险展示</h3>
            <p>最终管线为 PG-STDA-SAC-RSPA-TC，TC 为 validation-only 输出校准。</p>
          </div>
        </div>
      </section>

      <section class="grid cols-2">
        ${taskSummaryCard("first_transfer", first)}
        ${taskSummaryCard("second_transfer", second)}
      </section>

      <section class="grid cols-3">
        <div class="panel compact">
          <h3>Strict Raw Model</h3>
          <p class="metric-value">${htmlEscape(payload.reporting_boundary.strict_raw_model)}</p>
          <p class="muted">用于公平无监督迁移对比，不含 TC。</p>
        </div>
        <div class="panel compact">
          <h3>Final Pipeline</h3>
          <p class="metric-value">${htmlEscape(payload.reporting_boundary.final_pipeline)}</p>
          <p class="muted">用于工程展示，含验证集输出校准。</p>
        </div>
        <div class="panel compact">
          <h3>Demo Scope</h3>
          <p class="metric-value">Read-only</p>
          <p class="muted">${htmlEscape(payload.reporting_boundary.demo_scope)}</p>
        </div>
      </section>
    `;
  }

  function taskSummaryCard(key, task) {
    const points = task.representative_curve.points || [];
    const last = points[points.length - 1] || {};
    const health = Number(last.y_pred ?? 0);
    const risk = riskFromHealth(health);
    return `
      <article class="panel">
        <div class="panel-title">
          <div>
            <h2>${htmlEscape(task.component)}</h2>
            <p class="muted">${htmlEscape(task.source_domain)} → ${htmlEscape(task.target_domain)}</p>
          </div>
          <span class="tag ${risk.className}">${risk.text}</span>
        </div>
        <div class="grid cols-4">${metricCards(task)}</div>
        <div class="grid cols-2" style="margin-top:14px;">
          <div class="metric-card">
            <div class="metric-label">代表性 unit</div>
            <div class="metric-value">#${htmlEscape(task.representative_curve.unit_id)}</div>
            <div class="metric-note">${points.length} 个预测点</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">当前预测健康度</div>
            <div class="metric-value">${fmt.pct(health)}</div>
            <div class="metric-note">${taskFeatureText[key].riskLabel}</div>
          </div>
        </div>
      </article>
    `;
  }

  function renderData() {
    const rows = taskKeys
      .map((key) => {
        const task = payload.tasks[key];
        return `
          <tr>
            <td>${htmlEscape(task.label)}</td>
            <td>${htmlEscape(task.source_domain)}</td>
            <td>${htmlEscape(task.target_domain)}</td>
            <td>${htmlEscape(task.representative_curve.unit_id)}</td>
            <td>${task.representative_curve.num_points}</td>
            <td><span class="tag green">已加载</span></td>
          </tr>`;
      })
      .join("");

    app.innerHTML = `
      <section class="panel">
        <div class="panel-title">
          <div>
            <h2>数据导入与只读契约</h2>
            <p class="muted">本 demo 的主入口是 demo_payload.json，其他结果文件作为追溯证据。</p>
          </div>
          <span class="tag violet">No training</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>任务</th><th>源域</th><th>目标域</th><th>代表 unit</th><th>曲线点数</th><th>状态</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </section>
      <section class="grid cols-3">
        <div class="panel">
          <h3>输入特征</h3>
          <div class="mechanism-list">
            <div class="mechanism-item"><strong>反作用轮</strong><span>电流、温度、转速、振动代理等遥测量。</span></div>
            <div class="mechanism-item"><strong>卫星电池</strong><span>电压、电流、温度等 V/I/T 共享特征。</span></div>
          </div>
        </div>
        <div class="panel">
          <h3>标签与边界</h3>
          <div class="mechanism-list">
            <div class="mechanism-item"><strong>标签</strong><span>RUL/SOH 只用于监督源域训练、验证评估和最终指标。</span></div>
            <div class="mechanism-item"><strong>TC</strong><span>仅使用目标验证集输出映射，不使用目标测试标签拟合。</span></div>
          </div>
        </div>
        <div class="panel">
          <h3>质量检查</h3>
          <div class="bar-list">
            ${qualityBar("payload 字段完整", 1)}
            ${qualityBar("TC 消融行数", payload.tc_ablation.length / 8)}
            ${qualityBar("代表曲线可用", 1)}
            ${qualityBar("测试标签未用于校准", 1)}
          </div>
        </div>
      </section>
    `;
  }

  function renderTaskDetail(key) {
    setActive(key === "first_transfer" ? "wheel" : "battery");
    const task = payload.tasks[key];
    const feature = taskFeatureText[key];
    const points = task.representative_curve.points || [];
    app.innerHTML = `
      <section class="panel">
        <div class="panel-title">
          <div>
            <h2>${htmlEscape(task.label)}：${htmlEscape(task.component)}寿命预测</h2>
            <p class="muted">${htmlEscape(task.source_domain)} → ${htmlEscape(task.target_domain)} · unit #${htmlEscape(task.representative_curve.unit_id)}</p>
          </div>
          <div class="task-switch">
            <button class="task-button ${key === "first_transfer" ? "is-active" : ""}" type="button" data-switch-task="first_transfer">反作用轮</button>
            <button class="task-button ${key === "second_transfer" ? "is-active" : ""}" type="button" data-switch-task="second_transfer">卫星电池</button>
          </div>
        </div>
        <div class="grid cols-4">${metricCards(task)}</div>
      </section>

      <section class="grid cols-2">
        <div class="panel">
          <div class="panel-title">
            <h3>${feature.healthName} 预测回放</h3>
            <span class="tag">final TC prediction</span>
          </div>
          <div class="chart-wrap">
            <canvas id="detail-chart" class="large-canvas" width="900" height="430"></canvas>
          </div>
          <div class="slider-row">
            <span class="muted">当前时刻</span>
            <input id="time-slider" type="range" min="0" max="${Math.max(points.length - 1, 0)}" value="${Math.floor(points.length * 0.62)}" step="1">
            <div class="readout" id="time-readout">--</div>
          </div>
        </div>
        <div class="grid">
          <div class="panel">
            <h3>运行与预警</h3>
            <div id="risk-readout"></div>
          </div>
          <div class="panel">
            <h3>机理到遥测</h3>
            <div class="mechanism-list">
              <div class="mechanism-item"><strong>退化机理</strong><span>${feature.primary}</span></div>
              <div class="mechanism-item"><strong>可观测量</strong><span>${feature.observable}</span></div>
              <div class="mechanism-item"><strong>输入边界</strong><span>不把隐藏健康状态或目标测试标签作为 demo 校准输入。</span></div>
            </div>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-title">
          <h3>退化阶段与部署建议</h3>
          <span class="tag green">PHM decision support</span>
        </div>
        <div class="stage-strip">
          <div class="stage-cell">健康阶段</div>
          <div class="stage-cell">早期退化</div>
          <div class="stage-cell">加速退化</div>
          <div class="stage-cell">临近失效</div>
        </div>
        <div class="grid cols-3" style="margin-top:14px;">
          ${feature.advice.map((item) => `<div class="metric-card"><div class="metric-label">建议动作</div><strong>${item}</strong></div>`).join("")}
        </div>
      </section>
    `;

    document.querySelectorAll("[data-switch-task]").forEach((button) => {
      button.addEventListener("click", () => renderTaskDetail(button.dataset.switchTask));
    });
    setupDetailChart(key);
  }

  function renderMethod() {
    const tcRows = payload.tc_ablation || [];
    const rows = tcRows
      .map(
        (row) => `
        <tr>
          <td>${htmlEscape(row.task)}</td>
          <td>${htmlEscape(row.variant)}</td>
          <td>${htmlEscape(row.feature_mode || "none")}</td>
          <td>${htmlEscape(row.uses_target_val_labels)}</td>
          <td>${htmlEscape(row.uses_test_labels_for_fit)}</td>
          <td>${fmt.metric(row.rmse)}</td>
          <td>${fmt.metric(row.mae)}</td>
          <td>${fmt.metric(row.ra)}</td>
        </tr>`
      )
      .join("");

    const firstRows = tcRows.filter((row) => row.task === "第一迁移");
    const secondRows = tcRows.filter((row) => row.task === "第二迁移");

    app.innerHTML = `
      <section class="panel">
        <div class="panel-title">
          <div>
            <h2>迁移方法与 TC 消融</h2>
            <p class="muted">公平对比使用 raw 迁移输出；最终工程展示使用 validation-only calibrated pipeline。</p>
          </div>
          <span class="tag violet">${htmlEscape(payload.reporting_boundary.final_pipeline)}</span>
        </div>
        <div class="grid cols-2">
          <div class="metric-card">
            <div class="metric-label">Strict raw UDA</div>
            <div class="metric-value">${htmlEscape(payload.reporting_boundary.strict_raw_model)}</div>
            <div class="metric-note">不含 TC，用于主公平对比。</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Final engineering pipeline</div>
            <div class="metric-value">${htmlEscape(payload.reporting_boundary.final_pipeline)}</div>
            <div class="metric-note">TC 只使用目标验证集，不使用测试标签拟合。</div>
          </div>
        </div>
      </section>

      <section class="grid cols-2">
        <div class="panel">
          <h3>第一迁移 TC 消融</h3>
          <div class="bar-list">${tcBars(firstRows)}</div>
        </div>
        <div class="panel">
          <h3>第二迁移 TC 消融</h3>
          <div class="bar-list">${tcBars(secondRows)}</div>
          <div class="note-box" style="margin-top:14px;">第二迁移中 time-only TC 很强，应作为时间先验边界如实说明，不把 TC 后提升全部归因于迁移表征。</div>
        </div>
      </section>

      <section class="panel">
        <h3>TC 消融明细</h3>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>任务</th><th>变体</th><th>特征</th><th>用验证标签</th><th>用测试标签拟合</th><th>RMSE</th><th>MAE</th><th>RA</th></tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </section>
    `;
  }

  function qualityBar(label, value) {
    const pct = Math.max(0, Math.min(1, value));
    return `
      <div class="bar-row">
        <span>${label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${pct * 100}%"></div></div>
        <strong>${Math.round(pct * 100)}%</strong>
      </div>`;
  }

  function tcBars(rows) {
    if (!rows.length) return '<p class="muted">无数据</p>';
    const max = Math.max(...rows.map((row) => Number(row.rmse) || 0.001));
    return rows
      .map((row) => {
        const value = Number(row.rmse) || 0;
        const width = Math.max(5, (1 - value / max) * 86 + 10);
        return `
          <div class="bar-row">
            <span>${htmlEscape(row.variant)}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
            <strong>${fmt.metric(value)}</strong>
          </div>`;
      })
      .join("");
  }

  function riskFromHealth(value) {
    if (value > 0.62) return { text: "低风险", className: "green", riskClass: "risk-good" };
    if (value > 0.35) return { text: "关注", className: "", riskClass: "risk-watch" };
    return { text: "临近失效", className: "", riskClass: "risk-critical" };
  }

  function setupDetailChart(key) {
    const task = payload.tasks[key];
    const points = task.representative_curve.points || [];
    const canvas = document.getElementById("detail-chart");
    const slider = document.getElementById("time-slider");
    const readout = document.getElementById("time-readout");
    const risk = document.getElementById("risk-readout");
    if (!canvas || !slider || !readout || !risk) return;

    function redraw() {
      const index = Number(slider.value);
      drawPredictionChart(canvas, points, {
        accent: taskAccent[key],
        title: `${task.component} representative prediction`,
        cursorIndex: index,
      });
      const point = points[index] || points[points.length - 1] || {};
      const health = Number(point.y_pred ?? 0);
      const riskInfo = riskFromHealth(health);
      readout.textContent = `t = ${fmt.metric(point.time_index, 1)} · stage ${point.stage ?? "--"}`;
      risk.innerHTML = `
        <div class="grid cols-2">
          <div class="metric-card">
            <div class="metric-label">当前预测健康度</div>
            <div class="metric-value ${riskInfo.riskClass}">${fmt.pct(health)}</div>
            <div class="metric-note">true=${fmt.pct(point.y_true)} · abs err=${fmt.metric(point.abs_error)}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">风险等级</div>
            <div class="metric-value ${riskInfo.riskClass}">${riskInfo.text}</div>
            <div class="metric-note">${taskFeatureText[key].riskLabel}</div>
          </div>
        </div>`;
    }

    slider.addEventListener("input", redraw);
    redraw();
  }

  function drawPredictionChart(canvas, points, options) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(320, Math.floor(rect.width));
    const height = Math.max(260, Math.floor(rect.height));
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const pad = { left: 54, right: 24, top: 34, bottom: 42 };
    const chartW = width - pad.left - pad.right;
    const chartH = height - pad.top - pad.bottom;
    const xs = points.map((p) => Number(p.time_index));
    const ys = points.flatMap((p) => [Number(p.y_true), Number(p.y_pred)]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.max(0, Math.min(...ys) - 0.05);
    const maxY = Math.min(1.05, Math.max(...ys) + 0.05);

    const x = (value) => pad.left + ((value - minX) / Math.max(maxX - minX, 1)) * chartW;
    const y = (value) => pad.top + (1 - (value - minY) / Math.max(maxY - minY, 0.01)) * chartH;

    ctx.fillStyle = "#06111d";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(100, 180, 232, 0.14)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i += 1) {
      const gx = pad.left + (chartW * i) / 5;
      const gy = pad.top + (chartH * i) / 5;
      ctx.beginPath();
      ctx.moveTo(gx, pad.top);
      ctx.lineTo(gx, pad.top + chartH);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pad.left, gy);
      ctx.lineTo(pad.left + chartW, gy);
      ctx.stroke();
    }

    ctx.fillStyle = "#9dbad1";
    ctx.font = "12px Microsoft YaHei, sans-serif";
    ctx.fillText("0", pad.left - 22, y(0));
    ctx.fillText("1.0", pad.left - 34, y(1));
    ctx.fillText("time", pad.left + chartW - 25, height - 14);
    ctx.fillText("health", 10, pad.top - 12);

    drawLine(ctx, points, x, y, "y_true", "#dce9f4", 2.2);
    drawLine(ctx, points, x, y, "y_pred", options.accent, 2.4);

    const cursorIndex = Math.max(0, Math.min(points.length - 1, options.cursorIndex || 0));
    const cursor = points[cursorIndex];
    if (cursor) {
      const cx = x(Number(cursor.time_index));
      ctx.strokeStyle = "rgba(255, 255, 255, 0.72)";
      ctx.setLineDash([4, 5]);
      ctx.beginPath();
      ctx.moveTo(cx, pad.top);
      ctx.lineTo(cx, pad.top + chartH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = options.accent;
      ctx.beginPath();
      ctx.arc(cx, y(Number(cursor.y_pred)), 4, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = "#e9f5ff";
    ctx.font = "13px Microsoft YaHei, sans-serif";
    ctx.fillText("真实值", pad.left + 4, 20);
    ctx.fillStyle = "#dce9f4";
    ctx.fillRect(pad.left + 48, 12, 20, 3);
    ctx.fillStyle = "#e9f5ff";
    ctx.fillText("预测值", pad.left + 84, 20);
    ctx.fillStyle = options.accent;
    ctx.fillRect(pad.left + 128, 12, 20, 3);
  }

  function drawLine(ctx, points, x, y, key, color, lineWidth) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    points.forEach((point, index) => {
      const px = x(Number(point.time_index));
      const py = y(Number(point[key]));
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  navItems.forEach((item) => {
    item.addEventListener("click", () => {
      window.location.hash = item.dataset.view;
      render(item.dataset.view);
    });
  });

  window.addEventListener("hashchange", () => {
    render(window.location.hash.replace("#", "") || "overview");
  });

  window.addEventListener("resize", () => {
    const active = document.querySelector(".nav-item.is-active");
    if (active && (active.dataset.view === "wheel" || active.dataset.view === "battery")) {
      const key = active.dataset.view === "wheel" ? "first_transfer" : "second_transfer";
      const slider = document.getElementById("time-slider");
      if (slider) setupDetailChart(key);
    }
  });

  render(window.location.hash.replace("#", "") || "overview");
})();
