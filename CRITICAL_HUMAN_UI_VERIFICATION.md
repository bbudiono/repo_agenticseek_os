# 🔍 HUMAN UI VERIFICATION GUIDE

**CRITICAL: Every step below MUST work for real humans testing the app**

## Step 1: Application Startup Verification

**Priority:** CRITICAL - Must Pass **[CRITICAL]**

### Actions to Perform:
1. Open terminal and cd to: frontend/agentic-seek-copilotkit-broken/
2. Run: npm start
3. Wait for "Local: http://localhost:3000" message
4. Open browser to http://localhost:3000
5. Verify app loads without errors

### Expected Result:
✅ App loads showing Dashboard tab with blue header "AgenticSeek - AI Multi-Agent Platform"

---

## Step 2: Dashboard Functionality Check

**Priority:** CRITICAL - Must Pass **[CRITICAL]**

### Actions to Perform:
1. Verify Dashboard tab is active (blue background)
2. Check 4 stat cards are visible: Total Agents, Total Tasks, System Load, Memory Usage
3. Verify numbers are not 0 or placeholder
4. Check Recent Activity section shows actual tasks
5. Click "Refresh Data" button in header

### Expected Result:
✅ Real statistics (agents: 3, tasks: 3), progress bars show percentages, refresh button works

---

## Step 3: Agent CRUD Operations Test

**Priority:** CRITICAL - Must Pass **[CRITICAL]**

### Actions to Perform:
1. Click "Agents" tab
2. Verify "AI Agents (3)" title and 3 agent cards visible
3. Click "Create Agent" button
4. Fill form: Name="Test Agent", Type="research", Description="Test description"
5. Click "Create Agent" submit button
6. Look for success message
7. Verify new agent appears in list
8. Click "Delete" on any agent
9. Confirm deletion in popup
10. Verify agent disappears

### Expected Result:
✅ Form works, shows "Agent created successfully!", new card appears, deletion works with confirmation

---

## Step 4: Task CRUD Operations Test

**Priority:** CRITICAL - Must Pass **[CRITICAL]**

### Actions to Perform:
1. Click "Tasks" tab
2. Verify "Tasks (3)" title and 3 task cards visible
3. Click "Create Task" button
4. Fill form: Title="Test Task", Description="Test", select any agent, Priority="high"
5. Click "Create Task" submit button
6. Look for success message
7. Find a task with "PENDING" status
8. Click "Execute Task" button
9. Verify status changes to "RUNNING"

### Expected Result:
✅ Form works, agent dropdown populated, shows "Task created successfully!", execute button changes status

---

## Step 5: Settings Configuration Test

**Priority:** CRITICAL - Must Pass **[CRITICAL]**

### Actions to Perform:
1. Click "Settings" tab
2. Verify API Configuration section is visible
3. Check API Endpoint field shows URL
4. Check Agent Configuration dropdown has tier options
5. Click "Save Settings" button
6. Click "Test Connection" button

### Expected Result:
✅ Settings form loads, shows "Settings saved successfully!", test connection refreshes data

---

## Step 6: Error Handling Verification

**Priority:** CRITICAL - Must Pass **[CRITICAL]**

### Actions to Perform:
1. Go to Agents tab, click "Create Agent"
2. Try submitting form with empty name field
3. Go to Tasks tab, click "Create Task"
4. Try submitting without selecting agent
5. Check browser console for JavaScript errors

### Expected Result:
✅ Form validation shows "Please fill in all required fields", no JavaScript crashes

---

## ✅ SUCCESS CRITERIA
All 6 steps must pass completely. Any step failure indicates the app is not ready for human testing.

## ❌ FAILURE INDICATORS
- Blank screens or infinite loading
- Buttons that show alert("Add functionality")  
- Forms that don't validate or submit
- JavaScript errors in browser console
- Any UI element that doesn't work as described

## 🚨 CRITICAL REQUIREMENT
**NO FAKE FUNCTIONALITY** - Every button, form, and interaction must perform real operations, not placeholder alerts.
