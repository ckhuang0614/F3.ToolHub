# Modbus 設定指引

以下說明可協助你在本機執行 `F3.ToolHub` 時，連線到實體 PLC 或模擬器。

## 1. 連線實體 PLC
1. 於各環境的 `appsettings`（例如 `appsettings.json`）更新 `Modbus` 區塊，設定符合 PLC 的 `Host`、`Port`、`SlaveId` 與讀取範圍。
2. 在 `Registers` 陣列描述每個邏輯量測點，讓 API 能輸出具有型別與縮放後的資料：
   ```json
   "Registers": [
     { "Name": "line.temperature", "Offset": 0, "DataType": "UInt16", "Scale": 0.1, "Unit": "C" },
     { "Name": "line.energy", "Offset": 2, "DataType": "UInt32", "Scale": 1.0, "Unit": "Wh" },
     { "Name": "line.ready", "Offset": 3, "DataType": "Boolean", "BitIndex": 0 }
   ]
   ```
   - `Offset` 以動作的 `StartAddress` 為基準。
   - 支援的 `DataType`：`UInt16`、`Int16`、`UInt32`、`Int32`、`Float32`、`Boolean`（可搭配 `BitIndex`）。
   - `Scale` 套用於數值（例如 123 * 0.1 = 12.3）。
   - 選填 `MinValue` / `MaxValue` 可在超出範圍時把 `Points[n].Quality` 調為 `Low` / `High`，方便儀表板標示異常。
3. 若僅需手動觸發，得以停用背景輪詢：
   ```json
   "Polling": {
     "Enabled": false
   }
   ```
4. 依需求調整 `Resilience` 區塊，設定重試、退避與熔斷行為：
   ```json
   "Resilience": {
     "MaxRetryAttempts": 5,
     "BaseDelayMilliseconds": 500,
     "MaxDelayMilliseconds": 5000,
     "CircuitBreakerFailureThreshold": 5,
     "CircuitBreakerDurationSeconds": 30,
     "ConnectionPoolSize": 4
   }
   ```
5. 使用 `Output` 區塊啟用下游輸出：
   ```json
   "Output": {
     "File": { "Enabled": true, "Path": "data/plc-history.jsonl" },
     "Mqtt": { "Enabled": true, "Host": "broker", "Port": 1883, "Topic": "plc/line-1" }
   }
   ```
   - 檔案匯出為 JSON Lines，方便 ETL 或資料庫匯入。
   - MQTT 匯出使用 QoS 1，並套用所設定的 Topic；若無 broker 可保持停用。
6. 如需推播告警，透過 `Monitoring` 區塊設定 Webhook：
   ```json
   "Monitoring": {
     "AlertWebhooks": [
       { "enabled": true, "name": "noc", "url": "https://hooks.example/teams", "secretHeaderName": "X-Alert-Secret", "secret": "<token>" }
     ]
   }
   ```
   - Webhook 會收到完整的 `PlcAlert` 負載，可依需要加入標頭或簽章。
   - 儀表板亦可直接呼叫 `/api/tools/plc/alerts` 取得同樣資料。
7. 重新啟動服務並於每個請求附上有效 API Key，可再透過 `/api/tools/plc/health` 觀察 Modbus 連線池與失敗計數（本機預設為 `http://localhost:5281`，部署後請改用實際網域）：
   ```bash
   curl -H "X-Api-Key: <your-prod-key>" http://localhost:5281/api/tools/plc/health
   # 或
   curl -H "X-Api-Key: <your-prod-key>" https://<your-domain>/api/tools/plc/health
   ```
8. 使用 `/api/tools/plc/mappings` 系列端點維護映射，同樣需替換為實際位址：
   ```bash
   # 本機示例
   curl -H "X-Api-Key: <key>" http://localhost:5281/api/tools/plc/mappings

   # 新增或更新單一動作
   curl -X PUT -H "Content-Type: application/json" -H "X-Api-Key: <key>" \
        http://localhost:5281/api/tools/plc/mappings/read-registers \
        -d '{
          "name":"read-registers",
          "description":"讀取預設暫存器",
          "startAddress":0,
          "numberOfPoints":8,
          "registers":[
            { "name":"line.temperature","offset":0,"dataType":"UInt16","scale":0.1,"unit":"C","minValue":0,"maxValue":80 }
          ]
        }'
   ```
9. 透過下列端點監控告警與健康狀態（請以實際部署位址取代 `https://<your-domain>`；若在本機開發預設埠為 `http://localhost:5281`）：
   ```bash
   # 範例：本機開發環境
   curl -H "X-Api-Key: <key>" http://localhost:5281/api/tools/plc/alerts
   curl -H "X-Api-Key: <key>" http://localhost:5281/api/tools/plc/health

   # 正式環境請改為實際網域
   curl -H "X-Api-Key: <key>" https://<your-domain>/api/tools/plc/alerts
   curl -H "X-Api-Key: <key>" https://<your-domain>/api/tools/plc/health
   curl https://<your-domain>/healthz
   ```

## 2. 使用模擬器測試
1. 安裝 Modbus 模擬器（如 [ModbusPal](https://sourceforge.net/projects/modbuspal/)）或啟動 Python 伺服器：
   ```bash
   # 建議以 Python 3.10 虛擬環境執行 Web Simulator。
   python -m pip install pymodbus aiohttp
   python F3.ToolHub.Scripts/run_modbus_sim.py --json-file ./F3.ToolHub.Scripts/pymodbus_web_demo.json
   ```
2. 以 `Simulator` 環境啟動 API，以載入 `appsettings.Simulator.json`：
   ```bash
   dotnet run --environment Simulator
   ```
   - 若需在程式內優雅停止，可取得 `host` 物件後呼叫 `await host.StopAsync()`（或 `await host.WaitForShutdownAsync()`)，確保背景服務、`DisposeAsync()`、Serilog flush、Webhook sink 都完成：
     ```csharp
     var host = CreateHostBuilder(args).Build();
     await host.StartAsync();
     Console.CancelKeyPress += async (_, e) =>
     {
         e.Cancel = true;
         await host.StopAsync();
     };
     await host.WaitForShutdownAsync();
     ```
3. 透過使用者祕密設定暫時性的模擬器金鑰（參考 `docs/Security.md`），並確認背景輪詢可讀到模擬暫存器：
   ```bash
   curl -H "X-Api-Key: <sim-key-from-secrets>" http://localhost:5080/api/tools/plc/data
   ```
4. 若需要官方 Web Simulator，可直接使用倉庫內的 `F3.ToolHub.Scripts/pymodbus_web_demo.json`：
     ```bash
     python -m pymodbus.server.simulator.main ^
       --json_file .\F3.ToolHub.Scripts\pymodbus_web_demo.json ^
       --modbus_server server ^
       --modbus_device demo-device ^
       --http_host 0.0.0.0 ^
       --http_port 8081
     ```
     - `server` 會在 `0.0.0.0:1502` 開啟 Modbus TCP，與 `appsettings.Simulator.json` 相容。
     - Web UI 監聽 `http://localhost:8081`，可在瀏覽器調整暫存器或查看呼叫/日誌。
     - **pymodbus 3.6+ 補丁**：最新版 `pymodbus` 在 `Calls` 頁面會因 `function.function_code` 為 `int` 而噴錯，可在本機 venv (`venv310/Lib/site-packages/pymodbus/server/simulator/http_server.py`) 將 `build_html_calls()` 中產生 function list 的迴圈改成：
       ```python
       function_codes = ""
       for function in DecodePDU(True).list_function_codes():
           code = getattr(function, "function_code", function)
           label = getattr(function, "function_code_name", f"function code {code}")
           selected_attr = "selected" if code == self.call_monitor.function else ""
           function_codes += f"<option value={code} {selected_attr}>{label}</option>"
       ```
       只需替換 function list 那段，不影響其餘程式碼，即可讓 Web UI 在新版 `pymodbus` 上正常顯示。
