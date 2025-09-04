// Shannon Data Visualization
(function() {
  'use strict';

  // Helper function to get best baseline name
  function getBestBaseline(result) {
    const baselines = [
      { name: 'Cognition', f1: result.cg_f1 || 0 },
      { name: 'IsolationForest', f1: result.if_f1 || 0 },
      { name: 'OneClassSVM', f1: result.ocsvm_f1 || 0 },
      { name: 'LOF', f1: result.lof_f1 || 0 }
    ];
    
    const best = baselines.reduce((prev, current) => 
      current.f1 > prev.f1 ? current : prev
    );
    
    return best.name;
  }
  
  // Helper function to get best baseline F1 score
  function getBestBaselineF1(result) {
    return Math.max(
      result.cg_f1 || 0,
      result.if_f1 || 0,
      result.ocsvm_f1 || 0,
      result.lof_f1 || 0
    );
  }
  
  // Load and display evaluation results
  async function loadResults() {
    const container = document.getElementById('eval-table');
    if (!container) return;

    try {
      const response = await fetch('/datasets/eval_results.json');
      if (!response.ok) {
        container.innerHTML = '<p>Unable to load evaluation results.</p>';
        return;
      }

      const data = await response.json();
      
      // Transform data to expected format
      const transformedData = {
        results: data.results.map(r => ({
          dataset: r.file || r.dataset || 'Unknown',
          category: r.domain || r.category || 'security',
          shannon_f1: r.sh_f1 || r.shannon_f1 || 0,
          best_baseline: getBestBaseline(r),
          best_baseline_f1: getBestBaselineF1(r)
        }))
      };
      
      // Use only the repository source of truth (datasets/eval_results.json)
      displayResults(transformedData, container);
    } catch (error) {
      console.error('Error loading results:', error);
      container.innerHTML = '<p>Error loading evaluation results.</p>';
    }
  }

  function displayResults(data, container) {
    if (!data.results || !Array.isArray(data.results)) {
      container.innerHTML = '<p>No results data available.</p>';
      return;
    }

    // Create controls with filter for CIC-IDS-2018
    const controls = document.createElement('div');
    controls.className = 'table-controls';
    controls.innerHTML = `
      <label>
        <input type="checkbox" id="show-delta" /> Show ΔF1
      </label>
      <label>
        <input type="checkbox" id="highlight-breakthrough" checked /> Highlight Breakthroughs (F1 > 0.95)
      </label>
      <select id="filter-dataset">
        <option value="all">All Datasets</option>
        <option value="cicids2018">CIC-IDS-2018 Only</option>
        <option value="security">Security Only</option>
      </select>
      <select id="sort-by">
        <option value="dataset">Sort by Dataset</option>
        <option value="shannon">Sort by Shannon F1</option>
        <option value="delta">Sort by ΔF1</option>
      </select>
    `;
    container.appendChild(controls);

    // Create table
    const table = document.createElement('table');
    table.className = 'results-table';
    
    const thead = document.createElement('thead');
    thead.innerHTML = `
      <tr>
        <th>Dataset</th>
        <th>Category</th>
        <th>Shannon F1</th>
        <th>Best Baseline</th>
        <th>Baseline F1</th>
        <th class="delta-col" style="display:none">ΔF1</th>
        <th>Winner</th>
      </tr>
    `;
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    
    // Sort and display results
    let sortedResults = [...data.results];
    let filteredResults = [...data.results];
    
    function renderTable() {
      tbody.innerHTML = '';
      
      // Apply filter
      const filterValue = document.getElementById('filter-dataset')?.value || 'all';
      if (filterValue === 'cicids2018') {
        filteredResults = sortedResults.filter(r => 
          r.dataset.toLowerCase().includes('2018') || 
          r.dataset.toLowerCase().includes('cic-ids')
        );
      } else if (filterValue === 'security') {
        filteredResults = sortedResults.filter(r => 
          r.category?.toLowerCase() === 'security'
        );
      } else {
        filteredResults = [...sortedResults];
      }
      
      const highlightBreakthrough = document.getElementById('highlight-breakthrough')?.checked ?? true;
      
      filteredResults.forEach(result => {
        const row = document.createElement('tr');
        const delta = result.shannon_f1 - result.best_baseline_f1;
        const isWinner = result.shannon_f1 > result.best_baseline_f1;
        const isBreakthrough = result.shannon_f1 > 0.95;
        const isCICIDS2018 = result.dataset.includes('2018') || result.dataset.includes('CIC');
        
        // Determine row class
        let rowClass = '';
        if (isBreakthrough && highlightBreakthrough) {
          rowClass = 'breakthrough';
        } else if (isWinner) {
          rowClass = 'winner';
        }
        if (isCICIDS2018) {
          rowClass += ' cicids2018-dataset';
        }
        row.className = rowClass.trim();
        
        // Special highlighting for standout CIC-IDS-2018 results
        const isStandout = 
          (result.dataset.includes('Friday-16-02-2018') && result.shannon_f1 > 0.99) ||
          (result.dataset.includes('Friday-02-03-2018') && result.shannon_f1 > 0.97);
        
        row.innerHTML = `
          <td>${result.dataset}${isBreakthrough ? ' <span class="breakthrough-badge">★</span>' : ''}</td>
          <td>${result.category || 'Unknown'}</td>
          <td class="${isWinner ? 'shannon-win' : ''} ${isStandout ? 'standout-result' : ''}">${result.shannon_f1.toFixed(3)}</td>
          <td>${result.best_baseline}</td>
          <td>${result.best_baseline_f1.toFixed(3)}</td>
          <td class="delta-col" style="display:none">${delta > 0 ? '+' : ''}${delta.toFixed(3)}</td>
          <td>${isWinner ? '✓ Shannon' : result.best_baseline}</td>
        `;
        tbody.appendChild(row);
      });
    }

    renderTable();
    table.appendChild(tbody);
    container.appendChild(table);

    // Add chart if Chart.js is available
    if (typeof Chart !== 'undefined') {
      const chartContainer = document.createElement('div');
      chartContainer.className = 'chart-container';
      chartContainer.innerHTML = '<canvas id="results-chart"></canvas>';
      container.appendChild(chartContainer);
      
      createChart(data.results);
    }

    // Event handlers
    document.getElementById('show-delta').addEventListener('change', (e) => {
      const deltaCols = document.querySelectorAll('.delta-col');
      deltaCols.forEach(col => {
        col.style.display = e.target.checked ? '' : 'none';
      });
    });

    document.getElementById('highlight-breakthrough').addEventListener('change', () => {
      renderTable();
    });

    document.getElementById('filter-dataset').addEventListener('change', () => {
      renderTable();
    });

    document.getElementById('sort-by').addEventListener('change', (e) => {
      switch(e.target.value) {
        case 'shannon':
          sortedResults.sort((a, b) => b.shannon_f1 - a.shannon_f1);
          break;
        case 'delta':
          sortedResults.sort((a, b) => 
            (b.shannon_f1 - b.best_baseline_f1) - (a.shannon_f1 - a.best_baseline_f1)
          );
          break;
        default:
          sortedResults.sort((a, b) => a.dataset.localeCompare(b.dataset));
      }
      renderTable();
    });
  }

  function createChart(results) {
    const ctx = document.getElementById('results-chart');
    if (!ctx) return;

    // Sort by Shannon F1 (descending) without injecting or reordering specific datasets
    const sortedForChart = [...results].sort((a, b) => b.shannon_f1 - a.shannon_f1);

    const datasets = sortedForChart.map(r => r.dataset);
    const shannonScores = sortedForChart.map(r => r.shannon_f1);
    const baselineScores = sortedForChart.map(r => r.best_baseline_f1);
    
    // Create colors with special highlighting for CIC-IDS-2018 and breakthroughs
    const shannonColors = sortedForChart.map(r => {
      const isBreakthrough = r.shannon_f1 > 0.95;
      const isCICIDS2018 = r.dataset.includes('2018') || r.dataset.includes('CIC');
      
      if (isBreakthrough && isCICIDS2018) {
        return '#10B981'; // Success green for breakthrough CIC-IDS results
      } else if (isBreakthrough) {
        return '#0052E0'; // Bell Labs blue for other breakthroughs
      } else if (isCICIDS2018) {
        return '#4A90E2'; // Lighter blue for CIC-IDS datasets
      }
      return 'rgba(0, 82, 224, 0.7)'; // Default Bell Labs blue
    });
    
    const shannonBorderColors = sortedForChart.map(r => {
      const isBreakthrough = r.shannon_f1 > 0.95;
      const isCICIDS2018 = r.dataset.includes('2018') || r.dataset.includes('CIC');
      
      if (isBreakthrough && isCICIDS2018) {
        return '#10B981';
      } else if (isBreakthrough) {
        return '#0052E0';
      } else if (isCICIDS2018) {
        return '#4A90E2';
      }
      return '#0052E0';
    });

    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: datasets,
        datasets: [
          {
            label: 'Shannon',
            data: shannonScores,
            backgroundColor: shannonColors,
            borderColor: shannonBorderColors,
            borderWidth: 2
          },
          {
            label: 'Best Baseline',
            data: baselineScores,
            backgroundColor: 'rgba(156, 163, 175, 0.5)', // Gray-400
            borderColor: 'rgba(107, 114, 128, 1)', // Gray-500
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Shannon Performance Across Datasets',
            font: {
              size: 16,
              weight: 'bold'
            },
            color: '#1F2937'
          },
          legend: {
            display: true,
            position: 'top',
            labels: {
              generateLabels: function(chart) {
                const original = Chart.defaults.plugins.legend.labels.generateLabels;
                const labels = original.call(this, chart);
                
                // Add custom legend items for our color coding
                labels.push({
                  text: 'CIC-IDS Breakthrough (F1>0.95)',
                  fillStyle: '#10B981',
                  strokeStyle: '#10B981',
                  lineWidth: 2,
                  hidden: false
                });
                labels.push({
                  text: 'Other Breakthrough (F1>0.95)',
                  fillStyle: '#0052E0',
                  strokeStyle: '#0052E0',
                  lineWidth: 2,
                  hidden: false
                });
                
                return labels;
              }
            }
          },
          tooltip: {
            callbacks: {
              afterLabel: function(context) {
                const result = sortedForChart[context.dataIndex];
                const delta = result.shannon_f1 - result.best_baseline_f1;
                let label = `Δ: ${delta > 0 ? '+' : ''}${delta.toFixed(3)}`;
                
                if (result.shannon_f1 > 0.95) {
                  label += ' ★ BREAKTHROUGH';
                }
                if (result.dataset.includes('Friday-16-02-2018') || 
                    result.dataset.includes('Friday-02-03-2018')) {
                  label += ' 🎯 STANDOUT';
                }
                return label;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 1.0,
            title: {
              display: true,
              text: 'F1 Score',
              font: {
                weight: 'bold'
              }
            },
            grid: {
              color: 'rgba(0, 0, 0, 0.05)'
            },
            ticks: {
              callback: function(value) {
                if (value === 0.95) {
                  return '0.95 ← Breakthrough';
                }
                return value.toFixed(2);
              }
            }
          },
          x: {
            ticks: {
              autoSkip: false,
              maxRotation: 45,
              minRotation: 45,
              font: function(context) {
                const dataset = datasets[context.index];
                if (dataset && (dataset.includes('2018') || dataset.includes('CIC'))) {
                  return {
                    weight: 'bold',
                    size: 11
                  };
                }
                return {
                  size: 10
                };
              },
              color: function(context) {
                const dataset = datasets[context.index];
                if (dataset && (dataset.includes('2018') || dataset.includes('CIC'))) {
                  return '#0052E0';
                }
                return '#6B7280';
              }
            },
            grid: {
              display: false
            }
          }
        },
        interaction: {
          intersect: false,
          mode: 'index'
        },
        animation: {
          duration: 1000,
          easing: 'easeOutQuart'
        }
      }
    });
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', loadResults);
  } else {
    loadResults();
  }

  // Export for testing
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      loadResults,
      displayResults,
      createChart
    };
  }
})();
