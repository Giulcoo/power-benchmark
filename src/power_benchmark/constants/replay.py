TABLE_CSS = """
<style>
  .figures-row {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin: 16px;
  }

  .figure-section {
    flex: 1 1 800px;
    min-width: 600px;
    padding: 12px;
    border: 1px solid #ddd;
    border-radius: 8px;
  }

  .figure-section h2 {
    margin-top: 0;
    color: #333;
  }

  .net-table {
    border-collapse: collapse;
    font-size: 12px;
    margin-bottom: 16px;
    width: 100%;
  }
  .net-table th, .net-table td {
    border: 1px solid #ccc;
    padding: 4px 8px;
    text-align: right;
  }
  .net-table th {
    background: #f0f0f0;
    text-align: center;
  }
  .net-panel { display: none; }
  .net-panel.active { display: block; }
</style>
"""