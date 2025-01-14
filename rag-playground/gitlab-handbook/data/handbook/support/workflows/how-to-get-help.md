---
title: How to Get Help
category: Handling tickets
description: Workflow for Support Engineers on how to get help when working on a ticket.
---

## Getting Help on a Ticket

When [working on tickets](https://about.gitlab.com/handbook/support/workflows/working-on-tickets.html), collaboration is critical, especially when troubleshooting complex issues, or technical areas of focus that fall outside of your experience level. Asking for help means having a [low level of shame](https://about.gitlab.com/handbook/values/#low-level-of-shame), and also shows that you are putting the customer first because you are working towards resolving their problem.

### How to Get Help Workflow

If you are stuck on a ticket, the following workflow seeks to help Support Engineers realize and utilize all of the resources available to progress a ticket to resolution. This workflow lists some common resources, you can lean on to get the help you need.

**If you’re stuck on a ticket…..**

Identify what's causing you to get stuck. Some examples are:

- I don't have the right knowledge to progress this ticket.
- The customer's query is out of scope, but they expect us to resolve this.
- There is a deep technical issue which needs a development expert's consult.

Then consider these options to help unblock you. And remember that
[escalating to unblock](https://about.gitlab.com/handbook/values/#escalate-to-unblock)
is an operating principle of Results.

#### Ask in your SGG

Ask in [your group's Slack channel](/handbook/support/support-global-groups/#slack) for help. You might get all the help you
   need in responses right there, or you might open up the group's Zoom room
   for an impromptu pairing session to work on the ticket. And remember:

   1. Be sure to provide a link to the ticket
   1. Be specific about the help you need
      - For example: "Kubernetes Runner help needed: user is running into X
        error, logs are saying Y, and we've tried Z. What else could it be?"

#### Bring the ticket before a group of peers

Other Support Engineers are a great resource to help out with tickets. To get help from peers, you can try one or more of the following:

1. Attend crush or help sessions such as those noted below (see the [GitLab Support calendar](https://calendar.google.com/calendar/u/0?cid=Z2l0bGFiLmNvbV85YnMxNTllaHJjNXRxZ2x1cjg4ZGpiZDUxa0Bncm91cC5jYWxlbmRhci5nb29nbGUuY29t) for times):
    - AMER Senior SE Help Sessions
    - APAC/AMER or EMEA/AMER crush sessions
    - APAC or EMEA crush / collaboration sessions
    - Senior Support Office Hours (varying times)
1. Ask for help in one of the broader
   [Support Slack channels](https://about.gitlab.com/handbook/support/#slack).
1. Initiate a [Branch Out session](/handbook/support/support-global-groups/branch-out-sessions.html)
   - This is like a crush session, with the specific intention of helping 1 or 2
     groups with FRTs

#### Expand to Support Pods and other subject matter experts

You can also do one or more of the following:

1. See if there is a [Support Pod]({{< ref "working-with-pods" >}}) that covers the area your ticket is in and ask one of the Pod members for help.
1. Ask an expert within Support. You can check the [Skills by Subject](https://gitlab-com.gitlab.io/support/team/skills-by-subject.html)
   Support page to see who might have the skills to assist, or reach out
   to the [Support Stable Counterpart]({{< ref "support-stable-counterparts" >}})
   for the appropriate product area. Mention those
   people in the thread and in the ticket to let them know you think they can help.
1. [Request help from the relevant GitLab Development Team](#how-to-use-gitlabcom-to-formally-request-help-from-the-gitlab-development-team).
   Gather what information you have and fill in as much detail as possible for the dev team in the issue. To get more attention, you can post in the relevant group Slack channel with a message and link to the issue.
   If you don't get a response within the SLO, contact the listed engineering manager in the project readme.
   See [below](#how-to-use-gitlabcom-to-formally-request-help-from-the-gitlab-development-team) for more details.
   If you have a reproducible issue, then go straight to a bug issue in the appropriate [GitLab product tracker](https://gitlab.com/gitlab-org).

#### Bring the ticket to managers

Especially if you feel you're stalled on a ticket and need assistance identifying next steps:

1. Always feel free to reach out to any available manager (such as your manager, or the [Support Manager On-call](/handbook/support/workflows/support_manager-on-call.html)) in the `#spt_managers` channel. They will help you to determine next steps.
   - Avoid messages with no identified DRI for responding in `#spt_managers` as they can be missed or be a victim to the bystander effect.
1. [Open a STAR](/handbook/support/internal-support/support-ticket-attention-requests.html#submitting-a-support-ticket-attention-request-star--starring-a-ticket) in situations where getting help is urgent and important because:
   - the customer has expressed unhappiness with the service we're delivering via the ticket
   - the support engineer has noticed a correlation between several of a customers tickets that could use a more cohesive response
   - there is an urgent need for action in a different region (for example, finding a ticket owner or scheduling a call)

### How to Use GitLab.com to Formally Request Help from the GitLab Development Team

Starting from `2022-06-13` the Support Team and the Development Team are rolling out a series of projects that will enable support engineers to request help from a GitLab Development group, for more information on this please review the [associated proposal](https://gitlab.com/gitlab-com/support/support-team-meta/-/issues/4023). The aim is to provide a formal and accountable workflow process for Support Engineers to request assistance from the various Development Sections for any technical issues which they are currently unable to progress.  Please note that this is an iterative process, which aims to roll out the process for each of the 10 development sections at GitLab. If the Development Section that you require assistance from is not listed in the table below then please continue to use the existing methods for contacting the relevant Development Teams, such as Slack.

#### How to find the correct Development Section and Group to reach out for help

The easiest way to determine the correct place for a Support Request for Help issue is to use the docs pages. One possible workflow is as follows:

1. Locate a documentation page for the feature or topic on which you need help.
1. Scroll down to the bottom of the page and click on either the "Edit this page" link.
1. This will open up the `.md` source file of that docs page, which contains both the `stage` and `group` responsible for it noted on the top.
1. Now go to the [Product Categories handbook page](https://about.gitlab.com/handbook/product/categories/) and search for the Development Section to which the group identified on the previous step belongs to.
1. Use the table and workflow below to create a Request for Help issue in the project identified above.

Alternatively, if you have set up the [Support dotfiles](https://gitlab.com/gitlab-com/support/toolbox/dotfiles), you can use the `gls_request_for_help` command to quickly retrieve the "New issue" link with the correct issue template.

*NOTE:* A video recording of a similar workflow as the one described above can be found in the [Support Training repository](https://gitlab.com/gitlab-com/support/support-training/-/blob/master/Support%20Specific%20Trainings/training-videos/Support%20Request%20for%20Help%20Workflow.mp4)

#### List of Development Sections and corresponding links to the Projects for requesting help

Note: Some sections are not yet available.
If a section is not listed, consider a confidential issue in the Gitlab project tracker and post in the relevant Slack channel.
Please follow [support epic #222](https://gitlab.com/groups/gitlab-com/support/-/epics/222) for more information.

| Development Section | Section Product and Group Breakdown | Link to the GitLab Project for requesting help |
| ----------- | -------------- | -------- |
| Ops (CI, CD) Section | [Ops Section Breakdown](https://about.gitlab.com/handbook/product/categories/#ops-section)| [Section Ops Request for Help](https://gitlab.com/gitlab-com/ops-sub-department/section-ops-request-for-help/) |
| Dev Section | [Dev Section Breakdown](https://about.gitlab.com/handbook/product/categories/#dev-section) | [Section Dev Request for Help](https://gitlab.com/gitlab-com/dev-sub-department/section-dev-request-for-help/) |
| Sec Section | [Sec Section Breakdown](https://about.gitlab.com/handbook/product/categories/#sec-section) | [Section Sec Request for Help](https://gitlab.com/gitlab-com/sec-sub-department/section-sec-request-for-help) |
| Enablement Section | [Enablement Section Breakdown](https://about.gitlab.com/handbook/product/categories/#enablement-section) | [Section Enablement Request for Help](https://gitlab.com/gitlab-com/enablement-sub-department/section-enable-request-for-help) |
| Fulfilment Section | [Fulfilment Section Breakdown](https://about.gitlab.com/handbook/product/categories/#fulfillment-section) | [Section Fulfilment Request for Help](https://gitlab.com/gitlab-com/fulfilment-sub-department/section-fulfilment-request-for-help) |
| SaaS Section (GitLab Dedicated) | [GitLab Dedicated Breakdown](https://about.gitlab.com/handbook/engineering/infrastructure/team/gitlab-dedicated/) | [Gitlab Dedicated Request for Help Template](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/issues/new?issuable_template=support_request) |

The steps for submitting a help request are as follows:

1. Review the above table and click the link to open the corresponding GitLab.com Development Section Project that you require help from.
1. Within each GitLab Project there is a `ReadMe` (displayed on the project page):
   1. Read the `Support Engineer User Guidance` section and follow the steps outlined.
   1. Read the `Development groups and their corresponding templates` section and use the Handbook links provided if you are unsure as to which Section Sub Group and corresponding template you should use.
1. Before submitting a new issue search the existing issues to check if a similar request has been made before.
1. If not, then submit a new issue to the project using the template you've identified.
   - The issue will be automatically assigned to the relevant developers for triaging.
1. If you encounter any problems, such as obtaining a timely response from Development, then please take the following steps.
   - Review the contacts listed in the issue template and check slack to see if they are on PTO. If they are on PTO, check slack for their backup and ping them via the issue otherwise ping the primary contact requesting an update via the issue.
   - Feel empowered to ping the corresponding Development Subgroup slack channel along with a link to the issue, requesting an update.
   - Make the corresponding Development Subgroup Engineering Manager aware via the issue. You can identify the relevant Engineering Manager by checking the Development Group Handbook Page from each Projects ReadMe Section which provides a section named `Development Groups with their corresponding templates and labels`.
   - Consider reaching out to a Support Engineering Manager for further guidance.
1. Lastly, prior to closing the issue please review the information within to determine if any of it can be used to update the GitLab documentation, if any of the information is a candidate to be considered for updating the GitLab documentation then add the label `documentation::candidate` so that the issue can be identifiable for future use. If you have actually updated the GitLab documentation then please add a link to the MR to the issue and add the label `documentation::created`.

**Please note:** GitLab Dedicated is the first iteration for implementing a Request For Help Section for the SaaS section of GitLab development, therefore at the moment the GitLab project structure and workflows may not be consistent with the other development sections in the above table. You can find out more information on GitLab Dedicated internal processes by visiting the [Dedicated Team's ReadMe](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/README.md).

### Quick Links and Resources

- [Needs Collaboration view](https://gitlab.zendesk.com/agent/filters/360080204660) in ZenDesk.
- Create a Support [pairing session issue](https://gitlab.com/gitlab-com/support/support-pairing).
- [Support Workflows](https://about.gitlab.com/handbook/support/workflows/) to follow relevant troubleshooting workflow.
- [Support Documentation links](https://about.gitlab.com/handbook/support/#documentation) for quick references to helpful GitLab documentation.
- [Skills by Subject](https://gitlab-com.gitlab.io/support/team/skills-by-subject.html) to find a Support Engineer scoped to the skill set needed for help.
- [DevOps Stages](https://about.gitlab.com/handbook/product/categories/#devops-stages) to find the right development or product team to reach out to.
- [Emergency runbooks](https://gitlab.com/gitlab-com/support/emergency-runbook/-/tree/master/.gitlab/issue_templates) with troubleshooting tips, even if not an emergency.
- See which [manager is on-call](https://gitlab.pagerduty.com/escalation_policies#PGNLUZ1) if guidance is needed on something urgent.

#### General Troubleshooting Resources

Every problem is a little bit different. Sometimes it makes sense to try a different troubleshooting technique. These resources talk about general purpose approaches to troubleshooting:

- Julia Evans' [comics](https://wizardzines.com/comics/), especially the ones about debugging
- [The Pocket Guide to Debugging (PDF)](https://store.wizardzines.com/products/the-pocket-guide-to-debugging-pdf)
- [General Purpose Troubleshooting Principles](https://brie.dev/troubleshooting/)
