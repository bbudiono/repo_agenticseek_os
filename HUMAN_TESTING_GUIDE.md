# 🧪 HUMAN TESTING GUIDE - AgenticSeek UI Verification

## Quick Start
1. Open terminal and navigate to: `frontend/agentic-seek-copilotkit-broken/`
2. Run: `npm start`
3. Open browser to: `http://localhost:3000`
4. Follow the test scenarios below

---

## Test 1: Dashboard Tab

### Steps to Follow:
1. Open the application in browser
2. Verify Dashboard tab is active by default
3. Check that 4 statistics cards are visible (Total Agents, Total Tasks, System Load, Memory Usage)
4. Verify progress bars show dynamic percentages
5. Check Recent Activity section shows task list
6. Click Refresh Data button and verify it works

### Expected Results:
✅ Dashboard loads immediately
✅ All statistics show real numbers (not 0 or placeholder)
✅ Progress bars are colored (green/orange/red based on values)
✅ Recent activity shows actual task names and timestamps
✅ Refresh button triggers data reload

---

## Test 2: Agents Tab

### Steps to Follow:
1. Click on "Agents" tab
2. Verify page shows "AI Agents (3)" with count
3. Click "Create Agent" button
4. Fill out the form with: Name="Test Agent", Type="research", Description="Test"
5. Click "Create Agent" submit button
6. Verify success message appears
7. Check new agent appears in the list
8. Click "Delete" button on any agent
9. Verify confirmation dialog appears
10. Confirm deletion and verify agent is removed

### Expected Results:
✅ Agents tab shows 3 existing agents with different statuses
✅ Create form has proper validation (required fields)
✅ Success message: "Agent 'Test Agent' created successfully!"
✅ New agent card appears immediately
✅ Deletion shows confirmation: "Are you sure you want to delete..."
✅ Agent disappears from list after deletion

---

## Test 3: Tasks Tab

### Steps to Follow:
1. Click on "Tasks" tab
2. Verify page shows "Tasks (3)" with count
3. Click "Create Task" button
4. Fill form: Title="Test Task", Description="Test", Select any agent, Priority="high"
5. Click "Create Task" submit button
6. Verify success message appears
7. Check new task appears in the list
8. Find a task with "PENDING" status
9. Click "Execute Task" button
10. Verify task status changes to "RUNNING"
11. Click "View Details" on any task

### Expected Results:
✅ Tasks tab shows 3 existing tasks with different statuses and priorities
✅ Create form has agent dropdown with actual agents
✅ Success message: "Task 'Test Task' created successfully!"
✅ New task card appears with correct agent assignment
✅ Execute button only appears on pending tasks
✅ Status changes immediately with timestamp update
✅ View Details shows complete task information

---

## Test 4: Settings Tab

### Steps to Follow:
1. Click on "Settings" tab
2. Verify API Configuration section is visible
3. Check API Endpoint field shows current URL
4. Verify Agent Configuration section exists
5. Check tier dropdown has options (Free, Pro, Business, Enterprise)
6. Click "Save Settings" button
7. Verify success message appears
8. Click "Test Connection" button
9. Verify it triggers data refresh

### Expected Results:
✅ Settings page loads with configuration forms
✅ API endpoint shows: http://localhost:8000/api or environment URL
✅ Tier dropdown works and shows all 4 options
✅ Save Settings shows: "Settings saved successfully!"
✅ Test Connection actually refreshes the data in other tabs

---

## Test 5: Error Handling

### Steps to Follow:
1. Try creating agent with empty name field
2. Try creating task without selecting agent
3. Check that error messages appear
4. Verify application doesn't crash on errors
5. Test that delete confirmations actually prevent accidental deletion

### Expected Results:
✅ Form validation prevents submission with: "Please fill in all required fields"
✅ No crashes or blank screens
✅ All operations have proper error handling
✅ Confirmation dialogs actually prevent data loss

---

## ✅ Success Criteria
- All tabs load without errors
- All buttons perform actual operations (no blank responses)
- Forms validate input and show appropriate messages
- CRUD operations work (Create, Read, Update, Delete)
- Real data appears in cards and statistics
- No placeholder text like "Add functionality here"

## ❌ Failure Indicators
- Blank screens or "Loading..." that never finishes
- Buttons that show alert("Add functionality") 
- Empty cards or statistics showing 0
- Forms that don't validate or submit
- Any JavaScript errors in browser console

## 🆘 If Tests Fail
1. Check browser console for JavaScript errors
2. Verify npm start is running without errors
3. Try refreshing the page (Ctrl+F5 or Cmd+Shift+R)
4. Report specific failing test number and observed behavior

**CRITICAL: Every element in this guide should work for real humans testing the app**
