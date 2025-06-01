/**
 * Frontend Fix for AgenticSeek React App
 * This patch addresses the "null is not an object (evaluating 'prev.blocks')" error
 */

// The issue is in App.js line 13: const [responseData, setResponseData] = useState(null);
// Change this to initialize with proper structure:

const [responseData, setResponseData] = useState({
    blocks: {},
    done: false,
    answer: "",
    agent_name: "",
    status: "waiting",
    uid: "",
    screenshot: null,
    screenshotTimestamp: null
});

// Alternative minimal fix: Add null check in fetchScreenshot function
// Change lines 51-56 from:
/*
return {
    ...prev,
    screenshot: imageUrl,
    screenshotTimestamp: new Date().getTime()
};
*/

// To:
/*
return {
    ...(prev || {}),
    screenshot: imageUrl,
    screenshotTimestamp: new Date().getTime()
};
*/

// And change lines 59-63 from:
/*
setResponseData((prev) => ({
    ...prev,
    screenshot: 'placeholder.png',
    screenshotTimestamp: new Date().getTime()
}));
*/

// To:
/*
setResponseData((prev) => ({
    ...(prev || {}),
    screenshot: 'placeholder.png', 
    screenshotTimestamp: new Date().getTime()
}));
*/