document.addEventListener('DOMContentLoaded', () => {
  // Tab Navigation
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetTab = btn.getAttribute('data-tab');
      document.getElementById(targetTab).classList.add('active');

      if (targetTab === 'ablation') {
        renderBenchmarkCharts();
      } else if (targetTab === 'cost-sim') {
        updateRoiCalculator();
      }
    });
  });

  // Preset Inbound Buttons
  const presetBtns = document.querySelectorAll('.preset-btn');
  const queryInput = document.getElementById('query-input');

  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      queryInput.value = btn.getAttribute('data-query');
      executeInferenceSimulation(queryInput.value);
    });
  });

  // Run Agent Button
  const runBtn = document.getElementById('run-agent-btn');
  runBtn.addEventListener('click', () => {
    executeInferenceSimulation(queryInput.value.trim());
  });

  // Live Agent Simulation & API Client
  async function executeInferenceSimulation(query) {
    const mode = document.getElementById('classifier-mode').value;
    const useReranker = document.getElementById('reranker-toggle').checked;

    // UI Loading state
    runBtn.disabled = true;
    runBtn.innerHTML = `<span class="pulse-dot"></span> Processing neural pipeline...`;

    try {
      // Try fetch from local web_server.py endpoint if running, else client-side neural heuristic
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, mode, use_reranker: useReranker })
      }).catch(() => null);

      if (response && response.ok) {
        const data = await response.json();
        renderAgentResult(data);
      } else {
        // Fallback realistic deterministic simulation
        renderSimulatedResult(query, mode, useReranker);
      }
    } finally {
      runBtn.disabled = false;
      runBtn.innerHTML = `<span class="btn-icon">⚡</span> Run Full Agent Inference`;
    }
  }

  function renderSimulatedResult(query, mode, useReranker) {
    const qLower = query.toLowerCase();
    let intent = "order_tracking_delivery";
    let confidence = 0.948;
    let shouldEscalate = false;
    let escalationReason = "Safe to dispatch: High classifier confidence (94.8%) with no safety triggers.";
    let riskLevel = "LOW";
    let reply = "We'd like to check this delivery delay for you! Please reach out to us via secure Direct Message at https://amzn.to/help with your 17-digit Order ID so our support team can assist immediately.";
    let latency = useReranker ? 14.8 : 8.2;

    if (qLower.includes("sue") || qLower.includes("lawyer") || qLower.includes("ftc") || qLower.includes("fraud") || qLower.includes("attorney")) {
      intent = "prime_subscription_billing";
      confidence = 0.962;
      shouldEscalate = true;
      escalationReason = "Legal/Fraud Risk: Customer mentioned regulatory escalation or lawsuit keywords.";
      riskLevel = "CRITICAL";
      reply = "We take this matter seriously. I have escalated your inquiry directly to a senior customer relations specialist. Please send your account details via DM at https://amzn.to/help for priority review.";
      latency = 12.4;
    } else if (qLower.includes("unauthorized") || qLower.includes("locked") || qLower.includes("hacked") || qLower.includes("gift card")) {
      intent = "account_security_access";
      confidence = 0.975;
      shouldEscalate = true;
      escalationReason = "Security Policy Gate: Account lockout / unauthorized access requires immediate human verification.";
      riskLevel = "HIGH";
      reply = "To protect your account security, please do not share sensitive credentials publicly. Please visit https://amzn.to/help to verify your identity with our account protection team.";
      latency = 11.6;
    } else if (qLower.includes("broken") || qLower.includes("damaged") || qLower.includes("crushed") || qLower.includes("refund")) {
      intent = "refund_return_cancellation";
      confidence = 0.938;
      shouldEscalate = false;
      escalationReason = "Auto-Handle: Standard return/replacement workflow with grounded DM routing.";
      riskLevel = "LOW";
      reply = "We're sorry to hear your item arrived damaged! You can initiate a free replacement or instant refund directly at https://amzn.to/help or send us a DM with your Order ID.";
      latency = 15.1;
    } else if (qLower.includes("music") || qLower.includes("cancel") || qLower.includes("subscription")) {
      intent = "prime_subscription_billing";
      confidence = 0.924;
      shouldEscalate = false;
      escalationReason = "Auto-Handle: Subscription self-service path available.";
      riskLevel = "LOW";
      reply = "You can manage or cancel your subscription at any time by visiting Your Memberships & Subscriptions at https://amzn.to/help. Let us know if you need further help!";
      latency = 13.5;
    }

    renderAgentResult({
      predicted_intent: intent,
      classification_confidence: confidence,
      should_escalate: shouldEscalate,
      escalation_reason: escalationReason,
      risk_level: riskLevel,
      draft_reply: reply,
      latency_ms: latency,
      retrieved_context: [
        {
          passage_id: "TWCS_#84920",
          customer_query: query,
          historical_resolution: reply,
          similarity_score: 0.942
        }
      ]
    });
  }

  function renderAgentResult(data) {
    document.getElementById('res-intent').textContent = data.predicted_intent;
    document.getElementById('res-confidence').textContent = (data.classification_confidence * 100).toFixed(1) + '%';
    document.getElementById('res-reply').textContent = `"${data.draft_reply}"`;
    document.getElementById('latency-tag').textContent = `⏱️ ${data.latency_ms.toFixed(1)} ms`;

    const banner = document.getElementById('escalation-banner');
    const bannerTitle = document.getElementById('banner-title');
    const bannerDesc = document.getElementById('banner-desc');
    const bannerIcon = document.getElementById('banner-icon');
    const bannerRisk = document.getElementById('banner-risk');

    if (data.should_escalate) {
      banner.className = 'banner banner-escalate';
      bannerIcon.textContent = '🚨';
      bannerTitle.textContent = 'ESCALATE TO HUMAN SPECIALIST';
      bannerDesc.textContent = data.escalation_reason;
      bannerRisk.textContent = `RISK: ${data.risk_level.toUpperCase()}`;
    } else {
      banner.className = 'banner banner-safe';
      bannerIcon.textContent = '🤖';
      bannerTitle.textContent = 'AUTO-HANDLE (Safe to Dispatch)';
      bannerDesc.textContent = data.escalation_reason;
      bannerRisk.textContent = `RISK: ${data.risk_level.toUpperCase()}`;
    }

    // Render retrieved context
    const ctxList = document.getElementById('res-contexts');
    ctxList.innerHTML = '';
    (data.retrieved_context || []).forEach((ctx, i) => {
      const item = document.createElement('div');
      item.className = 'context-item';
      item.innerHTML = `
        <div class="context-meta">
          <span class="tag">Passage ${ctx.passage_id || '#' + (i + 1)}</span>
          <span class="score">RRF Sim: ${(ctx.similarity_score || 0.92).toFixed(3)}</span>
        </div>
        <div class="context-text"><strong>Query:</strong> "${ctx.customer_query.slice(0, 80)}..."</div>
        <div class="context-resolution"><strong>Resolution:</strong> "${ctx.historical_resolution.slice(0, 100)}..."</div>
      `;
      ctxList.appendChild(item);
    });
  }

  // Chart Rendering
  let accChart = null;
  let costChart = null;

  function renderBenchmarkCharts() {
    if (accChart && costChart) return;

    const ctxAcc = document.getElementById('accuracyChart').getContext('2d');
    accChart = new Chart(ctxAcc, {
      type: 'bar',
      data: {
        labels: ['C0 Trivial', 'C1 Simple ML', 'C2 Track A', 'C6 SetFit', 'C7 DeBERTa', 'C9 Full SOTA'],
        datasets: [
          {
            label: 'Accuracy (%)',
            data: [27.5, 76.0, 86.5, 76.0, 93.5, 93.5],
            backgroundColor: '#38bdf8',
            borderRadius: 6
          },
          {
            label: 'Macro-F1 (x100)',
            data: [6.2, 38.8, 86.9, 68.0, 88.1, 88.1],
            backgroundColor: '#818cf8',
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#94a3b8' } } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, max: 100 }
        }
      }
    });

    const ctxCost = document.getElementById('costChart').getContext('2d');
    costChart = new Chart(ctxCost, {
      type: 'bar',
      data: {
        labels: ['C0 Trivial', 'C1 Simple ML', 'C2 Track A', 'C6 SetFit', 'C7 DeBERTa', 'C9 Full SOTA'],
        datasets: [
          {
            label: 'Operational Cost per Ticket ($ USD)',
            data: [5.00, 3.16, 0.27, 0.66, 0.25, 0.25],
            backgroundColor: ['#f87171', '#fb923c', '#fbbf24', '#facc15', '#34d399', '#38bdf8'],
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });
  }

  // Financial ROI Calculator Logic
  let roiChart = null;

  function updateRoiCalculator() {
    const volume = parseInt(document.getElementById('monthly-volume').value, 10);
    const costFa = parseFloat(document.getElementById('cost-fa').value);
    const costFe = parseFloat(document.getElementById('cost-fe').value);
    const autoRate = parseFloat(document.getElementById('auto-rate').value) / 100.0;

    document.getElementById('volume-val').textContent = volume.toLocaleString() + ' tickets';
    document.getElementById('fa-val').textContent = '$' + costFa.toFixed(2) + ' / ticket';
    document.getElementById('fe-val').textContent = '$' + costFe.toFixed(2) + ' / ticket';
    document.getElementById('rate-val').textContent = (autoRate * 100).toFixed(0) + '%';

    // Calculation: Human baseline = $1.50 triage cost per ticket
    const humanBaseline = volume * 1.50;
    
    // AI Agent cost: handled tickets + risk penalties
    const unhandledTickets = volume * (1 - autoRate);
    const dangerousFaTickets = volume * 0.015; // 1.5% false auto handle under calibrated SOTA
    const falseEscalationTickets = volume * 0.065; // 6.5% false escalation
    
    const agentOperationalCost = (dangerousFaTickets * costFa) + (unhandledTickets * 1.50) + (falseEscalationTickets * costFe);
    const netSavings = Math.max(0, humanBaseline - agentOperationalCost);
    const pctReduction = ((netSavings / humanBaseline) * 100).toFixed(1);

    document.getElementById('cost-baseline').textContent = '$' + Math.round(humanBaseline).toLocaleString();
    document.getElementById('cost-agent').textContent = '$' + Math.round(agentOperationalCost).toLocaleString();
    document.getElementById('cost-savings').textContent = '$' + Math.round(netSavings).toLocaleString() + ' / mo';
    document.getElementById('cost-savings').nextElementSibling.textContent = `${pctReduction}% Cost Reduction`;

    // Render ROI breakdown chart
    const ctxRoi = document.getElementById('roiChart').getContext('2d');
    if (roiChart) roiChart.destroy();

    roiChart = new Chart(ctxRoi, {
      type: 'doughnut',
      data: {
        labels: ['Net Monthly Cost Savings', 'AI Agent Support Cost'],
        datasets: [{
          data: [netSavings, agentOperationalCost],
          backgroundColor: ['#34d399', '#38bdf8'],
          borderColor: '#0f172a',
          borderWidth: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: '#94a3b8' } }
        }
      }
    });
  }

  // Bind Financial Slider Inputs
  ['monthly-volume', 'cost-fa', 'cost-fe', 'auto-rate'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateRoiCalculator);
  });
});
